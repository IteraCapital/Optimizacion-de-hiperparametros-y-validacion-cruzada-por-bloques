
import Models
import OptimizationTools
import Select_dataset
import numpy as np

df = Select_dataset.dataset
filtered_dataset= Select_dataset.ANOVA_importance(df,0.79,'Label')
filtered_dataset = Select_dataset.dateToStr(filtered_dataset)
y= filtered_dataset[["Label"]]
x = filtered_dataset.loc[:,filtered_dataset.columns!='Label']
# x = x.loc[:,x.columns!='exp_CCI']
# x = x.loc[:,x.columns!='exp2_CCI']
x_train = x.loc[(x.index>'2017-01-20')&(x.index<'2020-03-20')].reset_index(drop=True)
y_train = y.loc[(y.index>'2017-01-20')&(y.index<'2020-03-20')].reset_index(drop=True)

x_test = x.loc[(x.index>'2020-03-21')&(x.index<'2020-06-21')].reset_index(drop=True)
y_test = y.loc[(y.index>'2020-03-21')&(y.index<'2020-06-21')].reset_index(drop=True)

def normalize(column):
    mean = np.mean(column)
    std = np.std(column)
    column = list((i-mean)/std for i in column)
    return column

for i in x_train.columns:
    x_train[i]=normalize(x_train[i])
    x_test[i]=normalize(x_test[i])


param_dict ={
    'learning_rate': {
                    'start':1,
                   'stop':10,
                    'step':1,
                   'scale':10,
                   },
    'neuron percentage': { ##capas ocultas
                    'start':1,
                    'stop': 10,
                    'step':1,
                    'scale':10
                            },
    'layer percentage':{
                    'start':1,
                    'stop':60,
                    'step':1,
                    'scale':10
                        },
    'batch size':{
                'start':1,
                'stop':100,
                'step':10,
                'scale':1
                        }
}

n_particles = 30
iter = 3










model = OptimizationTools.optimizeNN(param_dict,50,10,x_train,y_train)
nmodel = Models.createNN(model["neuron percentage"],model["learning_rate"])#
#
#
#
#
#
#

#
# path = 'C:/Users/anuno/OneDrive/Documents/ITESO/PAP 2/'
# red_wine = pd.read_csv(path+'winequality-red.csv',sep=',')
# white_wine=pd.read_excel(path+'white_wine.xlsx')
#
# red_wine["type"] = 1
# white_wine["type"] = 0
# wines = [red_wine, white_wine]
# wines = pd.concat(wines,sort=False)
# y = np.ravel(wines.type)
#
# x = wines.loc[:,wines.columns!="type"]
# y = wines["type"]
# x_train, x_test, y_train, y_test_ = train_test_split(x, y, test_size=0.33)
#
#
#
# param = OptimizationTools.optimizeNN(optimizer,5,3,x_train,y_train)