import pandas as pd
import numpy as np
import os.path as path
import seaborn as sns; sns.set(style="white")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools

# defne directory where the python file resides
dir_path = path.dirname(path.realpath(__file__))
data_path = dir_path + "/data/"

def _load_base_turnout():
    base_turnout_time = pd.read_csv(data_path + "turnout_time_master.csv")
    return base_turnout_time
def _create_oa_areas():
    output_areas = pd.read_csv(data_path + "oa_master.csv")
    cas_rate = pd.read_csv(data_path + "base_cas_master.csv")
    m = pd.merge(output_areas, cas_rate,  how='left', left_on=['oa_code'], right_on = ['oa_code'])
    m.columns = ['oa_code', 'hour', 'dwelling_cas_rate', 'rtc_cas_rate']
    return m
def _create_drive_time():
    drive_time = pd.read_csv(data_path + "drive_time_master.csv")
    hours = pd.DataFrame({'hour': list(range(0, 24)), 'key': 0})
    drive_time['key'] = 0
    drive_time = drive_time.merge(hours, how='outer', on='key').drop(columns=['key'])
    return drive_time
def _create_turnout_scenarios(scenario_dict):
    for k, v in scenario_dict.items():
        if v == 'off':
            scenario_dict[k] = list(itertools.repeat(999,24))
        if v == 'wt':
            scenario_dict[k] = list(itertools.repeat(2,24))
        if v == 'rds':
            scenario_dict[k] = list(itertools.repeat(5,24))
        if v == 'daycrewed':
            scenario_dict[k] = list(itertools.repeat(5,np.nan)) + list(itertools.repeat(2,10)) + list(itertools.repeat(5,np.nan))
        if v == 'nightonly':
            scenario_dict[k] = list(itertools.repeat(5,np.nan)) + list(itertools.repeat(999,10)) + list(itertools.repeat(5,np.nan))
    return scenario_dict      
def _create_final_turnout(turnout_times_df, drive_time_df, scenario_name = None):
    r = pd.merge(drive_time_df, turnout_times_df,  how='left', left_on=['appliance_callsign', 'hour'], right_on = ['appliance_callsign', 'hour'])
    r['total_time'] = r.drive_time + r.turnout_time
    #if scenario != None:
    #    r = r[~r.appliance_callsign.isin(scenario)].copy()
    r['rank'] = r.groupby(['oa_code','hour'])['total_time'].rank('first', ascending=True)
    r = r[r['rank'] <= 2.0]
    r = pd.pivot_table(r, values = 'total_time', index=['oa_code', 'hour'], columns = 'rank').reset_index()
    r.columns = ['oa_code', 'hour', 'tt1', 'tt2']
    return r
def _create_plot(df):
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x="Increase in dwl risk %", y="Increase in rtc risk %", hue="scenario", data=df, alpha=.4, palette="muted", edgecolors='none', s=100)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], loc='upper right', frameon=False)
    ax.set(ylim=(-20, 20))
    ax.set(xlim=(-20, 20))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    plt.show()
def _calc_dwelling_fatalities(cas_rate, first_appliance_time, second_appliance_time):
    ratio_first_app = 0.72
    ratio_second_app = 0.28
    def _response_factor(appliance_response_time):
        return (0.0002 * (appliance_response_time ** 2)) - (0.0006 * appliance_response_time) + 0.0218
    r1 = cas_rate * ratio_first_app * _response_factor(first_appliance_time)
    r2 = cas_rate * ratio_second_app * _response_factor(second_appliance_time)
    return r1 + r2
def _calc_rtc_fatalities(cas_rate, first_appliance_time, second_appliance_time):
    a = first_appliance_time * 0.0024
    b = a + 0.0202
    c = b * 0.93
    d = second_appliance_time / first_appliance_time
    e = (0.026 * d) + 0.93
    f = c * e
    return cas_rate * f
def _calculate_scores(oa_areas_df, turnout_times_df, drive_time_df, scenario_name = None):
    r = _create_final_turnout(turnout_times_df, drive_time_df, scenario_name)
    df = oa_areas_df.merge(r, how='left', on=('oa_code','hour'))
    df['dwelling_score'] = _calc_dwelling_fatalities(df['dwelling_cas_rate'].values, df['tt1'].values, df['tt2'].values)
    df['rtc_score'] = _calc_rtc_fatalities(df['rtc_cas_rate'].values, df['tt1'].values, df['tt2'].values)
    df = df.aggregate({"dwelling_score":['sum'],"rtc_score":['sum']}).reset_index(drop=True) 
    df['scenario'] = df.apply(lambda x: scenario_name if scenario_name != None else 'Base Case', axis = 1)
    df = df[['scenario','dwelling_score','rtc_score']]
    return df
def _create_final_df(df):
    df['Add dwl fatals per 10 yrs'] = 10 * (df.dwelling_score - df.dwelling_score.iloc[0])
    df['Add rtc fatals per 10 yrs'] = 10 * (df.rtc_score - df.rtc_score.iloc[0])
    df['Years per addition dwl fatal'] = 10 / df['Add dwl fatals per 10 yrs'] 
    df['Years per addition rtc fatal'] = 10 / df['Add rtc fatals per 10 yrs']
    df['Increase in dwl risk %'] = ((df.dwelling_score / df.dwelling_score.iloc[0])-1)*100
    df['Increase in rtc risk %'] = ((df.rtc_score / df.rtc_score.iloc[0])-1)*100
    return df
def run_batch_scenarios(scenario_list = None):
    scenario_count = len([s[0] for s in scenario_list])
    print(f'...................... Running modeller for {scenario_count} scenarios ......................')
    o = _create_oa_areas()
    t = _load_base_turnout()
    d = _create_drive_time()
    b = _calculate_scores(o, t, d)
    print('Running base model')
    for scenario_name, scenario_dict in scenario_list:
        print(f'Running scenario: {scenario_name}')
        ts = pd.DataFrame.from_dict(_create_turnout_scenarios(scenario_dict), orient='index')
        ts.reset_index(level=0, inplace=True)
        ts.rename(columns={"index": "appliance_callsign"}, inplace=True)
        ts = ts.melt(id_vars=['appliance_callsign'], var_name='hour', value_name='turnout_time')
        new_t = pd.merge(t, ts,  how='left', left_on=['appliance_callsign', 'hour'], right_on = ['appliance_callsign', 'hour'])
        new_t['turnout_time'] = new_t['turnout_time_y'].mask(pd.isnull, new_t['turnout_time_x'])
        new_t.drop(['turnout_time_x', 'turnout_time_y'], axis=1, inplace=True)
        s = _calculate_scores(o, new_t, d, scenario_name)
        b = pd.concat([b, s])
    b = _create_final_df(b)
    i = 0
    while i < 8:
        if path.exists(f'{dir_path}/scenario_export{i}.csv') == True:
           i+=1
        b.to_csv(f'{dir_path}/scenario_export{i}.csv', index = None, header=True)
        i += 8
    #b.to_csv(f'{dir_path}/scenario_export.csv', index = None, header=True)
    print(f'csv successfully created at: {dir_path}/')
    _create_plot(b)    
    return  b.reset_index(drop=True)

# create batch scenarios to make pumps off, wt, rds, day crewed or night time only
scenario_list = list()
scenario_list.append(
    ('Remove Crownhill', {
    "KV49P1": "off",
    "KV49P2": "off"
    }))
scenario_list.append(
    ('Remove Taunton', {
    "KV61P1": "off",
    "KV61P2": "off",
    "KV61P5": "off"
    }))
scenario_list.append(
    ('Shut Plymouth', {
    "KV47P1": "off",
    "KV48P1": "off",
    "KV49P1": "off",
    "KV49P2": "off",
    "KV50P1": "off",
    "KV50P2": "off",
    "KV51P1": "off"
    }))
scenario_list.append(
    ('Remove Exeter + Topsham', {
    "KV32P1": "off",
    "KV32P2": "off",
    "KV45P1": "off",
    "KV45P2": "off",
    "KV59P1": "off"
    }))
#scenario_list.append(
#    ('Wellington WT', {
#    "KV70P1": "wt"
#    }))
#scenario_list.append(
#    ('North Tawton WT', {
#    "KV12P1": "wt"
#    }))

print(run_batch_scenarios(scenario_list))
