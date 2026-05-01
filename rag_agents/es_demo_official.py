from datetime import datetime
import os
from elasticsearch import Elasticsearch

es_url = os.environ.get("ELASTICSEARCH_URL", "https://localhost:9200/")
es_username = os.environ.get("ELASTICSEARCH_USERNAME")
es_password = os.environ.get("ELASTICSEARCH_PASSWORD")
basic_auth = (es_username, es_password) if es_username and es_password else None
client = Elasticsearch([es_url], basic_auth=basic_auth, verify_certs=False)

doc = {
    "author": "kimchy",
    "text": "Elasticsearch: cool. bonsai cool.",
    "timestamp": datetime.now(),
}
resp = client.index(index="test-index", id=1, document=doc)
print(resp["result"])

resp = client.get(index="test-index", id=1)
print(resp["_source"])

client.indices.refresh(index="test-index")

resp = client.search(index="test-index", query={"match_all": {}})
print("Got {} hits:".format(resp["hits"]["total"]["value"]))
for hit in resp["hits"]["hits"]:
    print("{timestamp} {author} {text}".format(**hit["_source"]))
