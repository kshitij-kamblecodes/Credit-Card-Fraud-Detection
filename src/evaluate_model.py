from sklearn.metrics import classification_report, confusion_matrix, recall_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Recall:", recall)
    print(report)
    print("Confusion Matrix:\n", matrix)

    return recall, report, matrix
