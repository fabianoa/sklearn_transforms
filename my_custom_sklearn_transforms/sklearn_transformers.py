from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data['CHECKING_BALANCE'] =  data['CHECKING_BALANCE'].apply(lambda x: 0 if x=='NO_CHECKING' else float(x))
        data['EXISTING_SAVINGS'] =  data['EXISTING_SAVINGS'].apply(lambda x: 0 if x=='UNKNOWN' else float(x))
        data['LOAN_AMOUNT_RATE'] =  data['EXISTING_SAVINGS']/data['LOAN_AMOUNT']
        
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class DeriveFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X
        data['TOTAL_REPROVACOES']=data['REPROVACOES_DE']+data['REPROVACOES_EM']+data['REPROVACOES_MF']+data['REPROVACOES_GO']
        data['MEDIA_NOTAS']=(data['NOTA_DE']+data['NOTA_EM']+data['NOTA_MF']+data['NOTA_GO'])/4
        data['ESCORE_PADRAO_DE']=(data['NOTA_DE']-data['NOTA_DE'].mean())/data['NOTA_DE'].std()
        data['ESCORE_PADRAO_EM']=(data['NOTA_EM']-data['NOTA_EM'].mean())/data['NOTA_EM'].std()
        data['ESCORE_PADRAO_MF']=(data['NOTA_MF']-data['NOTA_MF'].mean())/data['NOTA_MF'].std()
        data['ESCORE_PADRAO_GO']=(data['NOTA_GO']-data['NOTA_GO'].mean())/data['NOTA_GO'].std()
        data['ESCORE_PADRAO_MEDIA_NOTAS']=(data['MEDIA_NOTAS']-data['MEDIA_NOTAS'].mean())/data['MEDIA_NOTAS'].std()

        # Retornamos um novo dataframe sem as colunas indesejadas
        return data