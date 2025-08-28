import numpy as np
from sklearn.cluster import KMeans
import psycopg2

# simple kmeans clustering and update Postgres tags

def main():
    X = np.load("embeddings.npy")
    k = 128
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(X)

    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/legal_ai_db")
    cur = conn.cursor()
    for i, lbl in enumerate(labels):
        # assuming you have mapping from index -> event id
        event_id = str(i)
        cur.execute("UPDATE feedback_events SET meta = jsonb_set(coalesce(meta,'{}'), '{cluster}', to_jsonb(%s::int)) WHERE id = %s", (lbl, event_id))
    conn.commit()
    cur.close(); conn.close()

if __name__ == '__main__':
    main()
