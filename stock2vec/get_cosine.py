import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

with open('stocks_emb.vec') as f: # open vector file
    lines = f.readlines()
    a = lines
data = a[1:] # drop the first line with parameters
df = {}

for i in range(len(data)): # line i is a vector with name and values
    vec = data[i].split(" ")
    name = vec[0]
    vec = vec[1:-1]
    df[name] = vec

df = pd.DataFrame(df, dtype="float64")
df = df.T

# get cosine_similarity
cos = pd.DataFrame(cosine_similarity(df), index=df.index, columns= df.index)
cos.to_csv("cos.csv")

# import pandas as pd
# df = pd.read_csv("cos.csv", index_col= 0)
print(df.loc["AAPL","BANF"]) # get cosine_similarity of AAPL and BANF
print(df.loc["BANF","MSFT"])
print(df.loc["AAPL","FRBA"])
print(df.loc["FRBA","MSFT"])

print(df.loc["BANF","FRBA"])
print(df.loc["AAPL","MSFT"])