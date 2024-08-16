import pickle

import numpy as np
import json
import hdbscan
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from extract import extract_text_from_table_json
from open_metadata_service import get_document

new_data = '''
{
   "name":"all_apply_key_columns",
   "fullyQualifiedName":"Oracle.default.sys.all_apply_key_columns",
   "columns":[
      {
         "name":"object_owner",
         "dataType":"VARCHAR",
         "description":"Owner of the object"
      },
      {
         "name":"object_name",
         "dataType":"VARCHAR",
         "description":"Name of the object"
      },
      {
         "name":"column_name",
         "dataType":"VARCHAR",
         "description":"Column name of the object"
      },
      {
         "name":"apply_database_link",
         "dataType":"VARCHAR",
         "description":"Remote database link to which changes will be aplied"
      },
      {
         "name":"apply_name",
         "dataType":"VARCHAR",
         "description":"Name of the apply process"
      }
   ],
   "service":{
      "displayName":"Oracle"
   }
}
'''


def init_clustering():
    '''
    주기적으로 기존 데이터를 이용한 Clustering
    '''
    documents, fqns = get_document()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine')
    labels = hdbscan_clusterer.fit_predict(X)

    df = pd.DataFrame({'document': documents, 'fqn': fqns, 'labels': labels})
    df.to_csv('hdbscan_clusters.csv', index=False)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


def find_similar_data():
    '''
    Clustering된 데이터를 이용한 데이터 추천 기능
    '''
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    df_existing = pd.read_csv('hdbscan_clusters.csv')
    existing_documents = df_existing['document'].tolist()
    existing_fqn = df_existing['fqn'].tolist()

    new_document = [extract_text_from_table_json(json.loads(new_data))[0]]

    x_new = vectorizer.transform(new_document)
    top_n = 5

    for i, new_doc in enumerate(new_document):
        similarities = cosine_similarity(x_new[i], vectorizer.transform(existing_documents))
        most_similar_idx = np.argmax(similarities)
        top_indices = np.argsort(similarities.flatten())[-top_n:]

        print(f"new Document\n{new_doc}")
        print(f"\nMost Similar Existing Document\n{existing_documents[most_similar_idx]}")
        print(f"\nMost Similar Found Document's FQN\n{existing_fqn[most_similar_idx]}")
        print(f"\nTop Similar Exist Document's index: {list(map(int, reversed(top_indices)))}")
        print()


if __name__ == '__main__':
    init_clustering()
    find_similar_data()
