import pandas as pd

df = pd.read_csv('free_parameter_tab', sep='\t')
df

Name = []
Column = []
Free_p = []
for t in range(len(df)):
    for i in range(2):
        Name.append((df['Name'][t]))
        Column.append((df['column'][t]))
        if i == 0:
          Free_p.append((df['Min'][t]))
        else :
          Free_p.append((df['Max'][t]))

d = {'Column': Column, 'Name': Name, 'Free_p': Free_p}
df_out = pd.DataFrame(d)
df_out 

df_out.to_csv('free_parameter_list', header=False, index=False, sep = '\t')
