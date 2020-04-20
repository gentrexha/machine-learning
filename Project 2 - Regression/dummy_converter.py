import pandas as pd

df = pd.read_csv('StudentPerformance.shuf.test.csv')

df.info()

#df = df.join(pd.get_dummies(df.state.replace(['ACC','RST','REQ','CLO'],[None,None,None,None],regex =True), prefix='state'))

#df = df.join(pd.get_dummies(df.service.replace(['irc','radius','dhcp','snmp','ssl','pop3'],[None,None,None,None,None,None],regex =True), prefix='service'))

df = df.join(pd.get_dummies(df['school'], prefix='school'))
df = df.join(pd.get_dummies(df['sex'], prefix='sex'))
df = df.join(pd.get_dummies(df['address'], prefix='address'))
df = df.join(pd.get_dummies(df['famsize'], prefix='famsize'))
df = df.join(pd.get_dummies(df['Pstatus'], prefix='Pstatus'))
df = df.join(pd.get_dummies(df['Mjob'], prefix='Mjob'))
df = df.join(pd.get_dummies(df['Fjob'], prefix='Fjob'))
df = df.join(pd.get_dummies(df['reason'], prefix='reason'))
df = df.join(pd.get_dummies(df['guardian'], prefix='guardian'))
df = df.join(pd.get_dummies(df['schoolsup'], prefix='schoolsup'))
df = df.join(pd.get_dummies(df['famsup'], prefix='famsup'))
df = df.join(pd.get_dummies(df['paid'], prefix='paid'))
df = df.join(pd.get_dummies(df['activities'], prefix='activities'))
df = df.join(pd.get_dummies(df['nursery'], prefix='nursery'))
df = df.join(pd.get_dummies(df['higher'], prefix='higher'))
df = df.join(pd.get_dummies(df['internet'], prefix='internet'))
df = df.join(pd.get_dummies(df['romantic'], prefix='romantic'))

df.info()

df.head()

df.to_csv('StudentPerformance.shuf.test_dum.csv', index=False)
