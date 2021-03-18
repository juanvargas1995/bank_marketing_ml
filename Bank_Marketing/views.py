from django.shortcuts import render

# our home page view
def home(request):    
    return render(request, 'index.html')

# custom method for generating predictions using logistic regression
def getPredictionsLR(age,job,marital,education,default,housing,loan,contact,month,day,duration,campaign,pdays,previous,poutcome,evr,cpi,cci,e3mr,num_emp):
    import pickle
    import pandas as pd
    import numpy as np
    
    scaled = pickle.load(open("scaler.sav", "rb"))
    encoder = pickle.load(open("encoder.sav", "rb"))
    selector = pickle.load(open("selector.sav", "rb"))
    model = pickle.load(open("bank_marketing_lr.sav", "rb"))

    bank= pd.DataFrame({'age':[age],'job':job,'marital':marital,'education':education,'default':default,'housing':housing,'loan':loan,'contact':contact, 'month':month, 'day_of_week':day,'duration':[duration],'campaign':[campaign],'pdays':[pdays],'previous':[previous],'poutcome':poutcome,'emp.var.rate':[evr],'cons.price.idx':[cpi],'cons.conf.idx':[cci],'euribor3m':[e3mr], 'nr.employed':[num_emp],'y':'NaN'}, )
    print('', flush=True)

    #Transforma data to [0-1] scale. 
    bank[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = scaled.transform(bank[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])

    bank_cat =['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome' ]
    
    enc_df = pd.DataFrame(encoder.transform(bank[bank_cat]).toarray())
    bank = bank.join(enc_df)

    for i in bank_cat:
      bank = bank.drop([i],axis=1)
    
    cols = selector.get_support(indices=True)
    bank_final = bank.iloc[:,cols]

    prediction = model.predict(bank_final)
    print('', flush=True)

    print(prediction, flush=True)
    if prediction == 0:
        return "No, user wouldn't subscribe"
    elif prediction == 1:
        return "Yes, user would subscribe"
    else:
        return "error"

# custom method for generating predictions using decission tree
def getPredictionsDT(age,job,marital,education,default,housing,loan,contact,month,day,duration,campaign,pdays,previous,poutcome,evr,cpi,cci,e3mr,num_emp):
    import pickle
    import pandas as pd
    import numpy as np
    
    scaled = pickle.load(open("scaler.sav", "rb"))
    encoder = pickle.load(open("encoder.sav", "rb"))
    selector = pickle.load(open("selector.sav", "rb"))
    model = pickle.load(open("bank_marketing_dt.sav", "rb"))

    bank= pd.DataFrame({'age':[age],'job':job,'marital':marital,'education':education,'default':default,'housing':housing,'loan':loan,'contact':contact, 'month':month, 'day_of_week':day,'duration':[duration],'campaign':[campaign],'pdays':[pdays],'previous':[previous],'poutcome':poutcome,'emp.var.rate':[evr],'cons.price.idx':[cpi],'cons.conf.idx':[cci],'euribor3m':[e3mr], 'nr.employed':[num_emp],'y':'NaN'}, )
    print('', flush=True)

    #Transforma data to [0-1] scale. 
    bank[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = scaled.transform(bank[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])

    bank_cat =['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome' ]
    
    enc_df = pd.DataFrame(encoder.transform(bank[bank_cat]).toarray())
    bank = bank.join(enc_df)

    for i in bank_cat:
      bank = bank.drop([i],axis=1)
    
    cols = selector.get_support(indices=True)
    bank_final = bank.iloc[:,cols]

    prediction = model.predict(bank_final)
    print('', flush=True)

    print(prediction, flush=True)
    if prediction == 0:
        return "No, user wouldn't subscribe"
    elif prediction == 1:
        return "Yes, user would subscribe"
    else:
        return "error"

# custom method for generating predictions using neural network
def getPredictionsNN(age,job,marital,education,default,housing,loan,contact,month,day,duration,campaign,pdays,previous,poutcome,evr,cpi,cci,e3mr,num_emp):
    import pickle
    import pandas as pd
    import numpy as np
    
    scaled = pickle.load(open("scaler.sav", "rb"))
    encoder = pickle.load(open("encoder.sav", "rb"))
    selector = pickle.load(open("selector.sav", "rb"))
    model = pickle.load(open("bank_marketing_nn.sav", "rb"))

    bank= pd.DataFrame({'age':[age],'job':job,'marital':marital,'education':education,'default':default,'housing':housing,'loan':loan,'contact':contact, 'month':month, 'day_of_week':day,'duration':[duration],'campaign':[campaign],'pdays':[pdays],'previous':[previous],'poutcome':poutcome,'emp.var.rate':[evr],'cons.price.idx':[cpi],'cons.conf.idx':[cci],'euribor3m':[e3mr], 'nr.employed':[num_emp],'y':'NaN'}, )
    print('', flush=True)

    #Transforma data to [0-1] scale. 
    bank[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = scaled.transform(bank[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])

    bank_cat =['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome' ]
    
    enc_df = pd.DataFrame(encoder.transform(bank[bank_cat]).toarray())
    bank = bank.join(enc_df)

    for i in bank_cat:
      bank = bank.drop([i],axis=1)
    
    cols = selector.get_support(indices=True)
    bank_final = bank.iloc[:,cols]

    prediction = model.predict(bank_final)
    print('', flush=True)

    print(prediction, flush=True)
    if prediction == 0:
        return "No, user wouldn't subscribe"
    elif prediction == 1:
        return "Yes, user would subscribe"
    else:
        return "error"

# our result page view
def result(request):
    age = int(request.GET['age'])
    job = str(request.GET['job'])
    marital = request.GET['marital']
    education = request.GET['education']
    default = request.GET['default']
    housing = request.GET['housing']
    loan = request.GET['loan']
    contact = request.GET['contact']
    day = request.GET['day']
    month = request.GET['month']
    duration = int(request.GET['duration'])
    campaign = int(request.GET['campaign'])
    pdays = int(request.GET['pdays'])
    previous = int(request.GET['previous'])
    poutcome = request.GET['poutcome']

    evr = int(request.GET['previous'])
    cpi = int(request.GET['cpi'])
    cci = int(request.GET['cci'])
    e3mr = request.GET['e3mr']
    num_emp = int(request.GET['numemp'])

    resultLR = getPredictionsLR(age,job,marital,education,default,housing,loan,contact,month,day,duration,campaign,pdays,previous,poutcome,evr,cpi,cci,e3mr,num_emp)
    resultDT = getPredictionsDT(age,job,marital,education,default,housing,loan,contact,month,day,duration,campaign,pdays,previous,poutcome,evr,cpi,cci,e3mr,num_emp)
    resultNN = getPredictionsNN(age,job,marital,education,default,housing,loan,contact,month,day,duration,campaign,pdays,previous,poutcome,evr,cpi,cci,e3mr,num_emp)

    return render(request, 'result.html', {'resultlr':resultLR, 'resultdt':resultDT, 'resultnn':resultNN})
