## Evaluate the testing set 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def evaluate(X_test,y_test,model ):
    
#    model.evaluate(X_test, y_test)
    pred = model.predict(X_test)

#    binary_predictions = []
#    
#    for i in pred:
#        if i >= 0.5:
#            binary_predictions.append(1)
#        else:
#            binary_predictions.append(0) 
            
    accuracy_sc = accuracy_score(pred, y_test)
    precision_sc = precision_score(pred, y_test)
    recall_sc = recall_score(pred, y_test)
    print('Accuracy on testing set:', accuracy_sc)
    print('Precision on testing set:', precision_sc)
    print('Recall on testing set:', recall_sc)
    
    
  
## Confusion matrix 
#    matrix = confusion_matrix(binary_predictions,y_test,normalize='all')
#    plt.figure(figsize=(16, 10))
#    ax= plt.subplot()
#    sns.heatmap(matrix, annot=True, ax = ax)

# labels, title and ticks
#    ax.set_xlabel('Predicted Labels', size=20)
#    ax.set_ylabel('True Labels', size=20)
#    ax.set_title('Confusion Matrix', size=20) 
#    ax.xaxis.set_ticklabels([0,1], size=15)
#    ax.yaxis.set_ticklabels([0,1], size=15)
#    
    return accuracy_sc,precision_sc,recall_sc,pred
