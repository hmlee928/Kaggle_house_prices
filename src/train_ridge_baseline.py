import pandas as pd
from sklearn.linear_model import Ridge

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
test_id = test["Id"].copy()

X = train.drop(["Id", "SalePrice"], axis=1)
y = train["SalePrice"]

X = pd.get_dummies(X)
test = pd.get_dummies(test)

X, test = X.align(test, join="left", axis=1, fill_value=0)

X = X.fillna(0)
test = test.fillna(0)

model = Ridge()
model.fit(X, y)

preds = model.predict(test)

submission = pd.DataFrame({
    "Id": test_id,
    "SalePrice": preds
})

submission.to_csv("outputs/submission_ridge_baseline.csv", index=False)

print("Saved submission!")
