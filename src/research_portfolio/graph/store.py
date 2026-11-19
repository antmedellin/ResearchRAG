"""Neo4j access: department-scoped sync (safe to run one department at a time, in any order),
existing-entity loading for dedup, and read-back for Qdrant indexing.
Ownership: MEMBER_OF -> department run; contributor edges -> the profile that listed them (claims);
department_slugs -> always derived from the graph."""
import os
from datetime import datetime, timezone
from neo4j import GraphDatabase
from research_portfolio.common import from_props

LABEL = {"publication": "Publication", "grant": "Grant", "presentation": "Presentation"}
REL   = {"publication": "AUTHORED", "grant": "INVESTIGATOR_ON", "presentation": "PRESENTED"}
CONTRIB_RELS = "AUTHORED|INVESTIGATOR_ON|PRESENTED"
ENTITY_MATCH = "(n:Publication OR n:Grant OR n:Presentation)"

SCHEMA = [
    "CREATE CONSTRAINT dept_slug IF NOT EXISTS FOR (n:Department)   REQUIRE n.slug IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person)       REQUIRE n.person_id IS UNIQUE",
    "CREATE CONSTRAINT pub_id    IF NOT EXISTS FOR (n:Publication)  REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT grant_id  IF NOT EXISTS FOR (n:Grant)        REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT pres_id   IF NOT EXISTS FOR (n:Presentation) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ikey      IF NOT EXISTS FOR (n:IdentityKey)  REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT topic_id  IF NOT EXISTS FOR (n:Topic)        REQUIRE n.topic_id IS UNIQUE",
    "CREATE CONSTRAINT sponsor   IF NOT EXISTS FOR (n:Sponsor)      REQUIRE n.key IS UNIQUE",
    "CREATE INDEX pub_year   IF NOT EXISTS FOR (n:Publication) ON (n.year)",
    "CREATE INDEX grant_year IF NOT EXISTS FOR (n:Grant)       ON (n.year)",
]

def driver():
    return GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://neo4j:7687"),
                                auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))

DB = os.environ.get("NEO4J_DATABASE", "neo4j")

def run_q(drv, cypher, **params):
    return drv.execute_query(cypher, params, database_=DB).records

def ensure_schema(drv):
    for s in SCHEMA:
        run_q(drv, s)

# ---------------- dedup support ----------------
def load_existing(drv, kind):
    rows = run_q(drv, f"""
        MATCH (n:{LABEL[kind]})
        OPTIONAL MATCH (k:IdentityKey)-[:IDENTIFIES]->(n)
        RETURN n {{.*}} AS n, collect(k.key) AS keys""")
    out = []
    for r in rows:
        d = from_props(dict(r["n"]))
        d["identity_keys"] = r["keys"]
        out.append(d)
    return out

# ---------------- department sync ----------------
Q_DEPT = """
MERGE (d:Department {slug: $slug})
SET d.name = $name, d.institution = $institution, d.aliases = $aliases, d.last_run = $run"""

Q_PERSON = """
UNWIND $rows AS p
MERGE (n:Person {person_id: p.person_id})
SET n += p.props, n.in_roster = true
WITH n, p
MATCH (d:Department {slug: $slug})
MERGE (n)-[r:MEMBER_OF]->(d)
SET r.rank = p.rank, r.run = $run"""

Q_TOPICS_CLEAR = """
UNWIND $rows AS p
MATCH (n:Person {person_id: p.person_id})-[t:INTERESTED_IN]->()
DELETE t"""

Q_TOPICS = """
UNWIND $rows AS p
MATCH (n:Person {person_id: p.person_id})
UNWIND p.topics AS t
MERGE (x:Topic {topic_id: t.id}) SET x.name = coalesce(x.name, t.name), x.source = 'dm_interest'
MERGE (n)-[:INTERESTED_IN]->(x)"""

def q_entity(label):
    return f"""
UNWIND $rows AS x
MERGE (n:{label} {{id: x.id}})
SET n += x.props, n.last_run = $run
WITH n, x UNWIND x.keys AS key
MERGE (k:IdentityKey {{key: key}})
MERGE (k)-[:IDENTIFIES]->(n)"""

Q_SPONSOR = """
UNWIND $rows AS x
MATCH (g:Grant {id: x.id})
OPTIONAL MATCH (g)-[s:SPONSORED_BY]->()
DELETE s
WITH DISTINCT g, x WHERE x.sponsor_key IS NOT NULL
MERGE (sp:Sponsor {key: x.sponsor_key}) SET sp.name = coalesce(sp.name, x.sponsor)
MERGE (g)-[:SPONSORED_BY]->(sp)"""

def q_claim(label, rel):
    return f"""
UNWIND $rows AS c
MATCH (n:{label} {{id: c.entity_id}})
MERGE (p:Person {{person_id: c.person_id}})
  ON CREATE SET p.in_roster = false, p.display_name = c.name, p.dm_user_id = c.dm_user_id
MERGE (p)-[r:{rel}]->(n)
SET r.position = c.position, r.role = c.role,
    r.claims = [x IN coalesce(r.claims, []) WHERE NOT x STARTS WITH c.owner + '@'] + [c.owner + '@' + $run]"""

Q_DEPARTED = """
MATCH (p:Person)-[r:MEMBER_OF]->(:Department {slug: $slug}) WHERE r.run <> $run
DELETE r
WITH DISTINCT p WHERE NOT (p)-[:MEMBER_OF]->()
SET p.in_roster = false
RETURN collect(p.person_id) AS departed"""

Q_RELEASE_CLAIMS = f"""
MATCH ()-[r:{CONTRIB_RELS}]->()
WHERE any(x IN r.claims WHERE split(x, '@')[0] IN $owners AND NOT x ENDS WITH '@' + $run)
SET r.claims = [x IN r.claims WHERE NOT split(x, '@')[0] IN $owners OR x ENDS WITH '@' + $run]
WITH r WHERE size(r.claims) = 0
DELETE r"""

Q_ORPHANS = f"""
MATCH (n) WHERE {ENTITY_MATCH} AND NOT ()-[:{CONTRIB_RELS}]->(n)
OPTIONAL MATCH (k:IdentityKey)-[:IDENTIFIES]->(n)
WITH n, n.id AS id, collect(k) AS ks
FOREACH (k IN ks | DETACH DELETE k)
DETACH DELETE n
RETURN collect(id) AS removed"""

CLEANUP_MISC = [
    "MATCH (p:Person {in_roster: false}) WHERE NOT (p)--() DELETE p",
    "MATCH (t:Topic) WHERE NOT (t)--() DELETE t",
    "MATCH (s:Sponsor) WHERE NOT (s)--() DELETE s",
]

Q_PERSON_SLUGS = """
MATCH (p:Person)
OPTIONAL MATCH (p)-[:MEMBER_OF]->(d:Department)
WITH p, d.slug AS s ORDER BY s
WITH p, collect(DISTINCT s) AS slugs
WHERE coalesce(p.department_slugs, []) <> slugs
SET p.department_slugs = slugs
RETURN p.person_id AS id, slugs"""

Q_ENTITY_SLUGS = f"""
MATCH (n) WHERE {ENTITY_MATCH}
OPTIONAL MATCH (n)<-[:{CONTRIB_RELS}]-(:Person)-[:MEMBER_OF]->(d:Department)
WITH n, d.slug AS s ORDER BY s
WITH n, collect(DISTINCT s) AS slugs
WHERE coalesce(n.department_slugs, []) <> slugs
SET n.department_slugs = slugs
RETURN n.id AS id, slugs"""

Q_DEPT_STATS = """
MATCH (d:Department {slug: $slug})
OPTIONAL MATCH (p:Person)-[:MEMBER_OF]->(d)
WITH d, count(p) AS n SET d.faculty_count = n RETURN n"""

def sync_department(drv, dept, people, entities, claims):
    """
    dept     : {"slug","name","institution","aliases"}
    people   : [{"person_id","rank","props":{...},"topics":[{"id","name"}]}]   - this department's roster
    entities : {kind: [{"id","keys","props","sponsor","sponsor_key"}]}           - resolved + merged
    claims   : {kind: [{"entity_id","person_id","owner","name","dm_user_id","position","role"}]}
    """
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    def q(cypher, **p):
        return run_q(drv, cypher, run=run, slug=dept["slug"], **p)
    def batched(cypher, rows, size=500):
        for i in range(0, len(rows), size):
            q(cypher, rows=rows[i:i + size])

    ensure_schema(drv)
    q(Q_DEPT, name=dept["name"], institution=dept["institution"], aliases=dept.get("aliases", []))
    batched(Q_PERSON, people)
    batched(Q_TOPICS_CLEAR, people)
    batched(Q_TOPICS, people)
    for kind, label in LABEL.items():
        batched(q_entity(label), entities.get(kind, []))
        if kind == "grant":
            batched(Q_SPONSOR, entities.get(kind, []))
        batched(q_claim(label, REL[kind]), claims.get(kind, []))

    departed = q(Q_DEPARTED)[0]["departed"]
    owners = sorted({p["person_id"] for p in people} | set(departed))
    q(Q_RELEASE_CLAIMS, owners=owners)
    removed = q(Q_ORPHANS)[0]["removed"]
    for c in CLEANUP_MISC:
        q(c)
    changes = [(r["id"], r["slugs"]) for r in q(Q_PERSON_SLUGS)] + \
              [(r["id"], r["slugs"]) for r in q(Q_ENTITY_SLUGS)]
    count = q(Q_DEPT_STATS)[0]["n"]
    return {"run": run, "departed": departed, "removed_entities": removed,
            "slug_changes": changes, "faculty_count": count}

# ---------------- read-back for indexing ----------------
def read_people(drv, ids):
    rows = run_q(drv, """
        MATCH (p:Person) WHERE p.person_id IN $ids
        OPTIONAL MATCH (p)-[m:MEMBER_OF]->(d:Department)
        WITH p, collect(CASE WHEN d IS NULL THEN NULL ELSE {slug: d.slug, name: d.name, rank: m.rank} END) AS mem
        RETURN p {.*} AS p, mem,
               COUNT { (p)-[:AUTHORED]->() } AS pubs,
               COUNT { (p)-[:INVESTIGATOR_ON]->() } AS grants,
               COUNT { (p)-[:INVESTIGATOR_ON]->(:Grant {status: 'Funded'}) } AS funded,
               [(p)-[:AUTHORED]->(x) WHERE x.year IS NOT NULL | x.year] AS years""", ids=ids)
    out = []
    for r in rows:
        p = from_props(dict(r["p"]))
        p.update(departments=r["mem"], publication_count=r["pubs"], grant_count=r["grants"],
                 funded_grant_count=r["funded"], latest_pub_year=max(r["years"]) if r["years"] else None)
        out.append(p)
    return out

def read_entities(drv, ids):
    rows = run_q(drv, f"""
        MATCH (n) WHERE {ENTITY_MATCH} AND n.id IN $ids
        OPTIONAL MATCH (p:Person)-[r:{CONTRIB_RELS}]->(n)
        WITH n, collect(DISTINCT p.person_id) AS pids,
             collect(DISTINCT CASE WHEN p.in_roster THEN p.display_name END) AS faculty,
             collect(DISTINCT CASE WHEN r.role IN ['PI', 'Co-PI'] THEN p.person_id END) AS leads
        RETURN toLower(labels(n)[0]) AS kind, n {{.*}} AS n, pids, faculty, leads""", ids=ids)
    out = []
    for r in rows:
        x = from_props(dict(r["n"]))
        x.update(kind=r["kind"], person_ids=sorted(r["pids"]), faculty_names=sorted(r["faculty"]),
                 lead_person_ids=sorted(r["leads"]))
        out.append(x)
    return out
