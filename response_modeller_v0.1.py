import pandas as pd
import os.path as path
import itertools

# defne directory where the python file resides
dir_path = path.dirname(path.realpath(__file__))

def _create_turnout_scenarios(scenario_dict):
    """ 
    Convert the dictionary to a list of 24 integers for each appliance being put off the run.
    Parameters: 
    List of scenarios (dict): The dictionary of the scenario to run
    Returns: 
    List of 24 integers for each hour of the day for each appliance in the dictionary
    """
    for k, v in scenario_dict.items():
        if v == 'off':
            scenario_dict[k] = list(itertools.repeat(999,24))
    return scenario_dict      
def _create_final_turnout(turnout_times_df, drive_time_df, scenario_name = None):
    """ 
    Calc the final tunrout times for each OA area per hour for the scenario, i.e. making the tunout 999 for those in the scneario or running base model
    Parameters: 
    turnout times (df): The turnout times per hour per OA, either the base or the scenario
    drive times (df): The drive time per hour per output area for the first ten pumps
    scenario_name (str): name of the scenario or base model for the first run
    Returns: 
    dataframe of oa area, hour and the theoretical first / second appliance in attendance. Either base model or once the scnario pumps are removed.
    """
    r = pd.merge(drive_time_df, turnout_times_df,  how='left', left_on=['appliance_callsign', 'hour'], right_on = ['appliance_callsign', 'hour'])
    r['total_time'] = r.drive_time.values + r.turnout_time.values
    r['rank'] = r.groupby(['oa_code','hour'])['total_time'].rank('first', ascending=True)
    r = r[r['rank'] <= 2.0]
    r = pd.pivot_table(r, values = 'total_time', index=['oa_code', 'hour'], columns = 'rank').reset_index()
    r.columns = ['oa_code', 'hour', 'tt1', 'tt2']
    return r
def _calc_dwelling_fatalities(cas_rate, first_appliance_time, second_appliance_time):
    """ 
    Calc the dwelling risk
    Parameters: 
    cas rate (float): The rate to be applied
    First app (float): The time of first in attendance in decimalised mins
    Second app (float): The time of second in attendance in decimalised mins
    Returns: 
    Calculated lives lost
    """
    ratio_first_app = 0.72
    ratio_second_app = 0.28
    def _response_factor(appliance_response_time):
        return (0.0002 * (appliance_response_time ** 2)) - (0.0006 * appliance_response_time) + 0.0218
    r1 = cas_rate * ratio_first_app * _response_factor(first_appliance_time)
    r2 = cas_rate * ratio_second_app * _response_factor(second_appliance_time)
    return r1 + r2
def _calc_rtc_fatalities(cas_rate, first_appliance_time, second_appliance_time):
    """ 
    Calc the rtc risk
    Parameters: 
    cas rate (float): The rate to be applied
    First app (float): The time of first in attendance in decimalised mins
    Second app (float): The time of second in attendance in decimalised mins
    Returns: 
    Calculated lives lost
    """
    a = first_appliance_time * 0.0024
    b = a + 0.0202
    c = b * 0.93
    d = second_appliance_time / first_appliance_time
    e = (0.026 * d) + 0.93
    f = c * e
    return cas_rate * f
def _calculate_scores(oa_areas_df, turnout_times_df, drive_time_df, scenario_name = None):
    """ 
    Calc the scores dwel fire and rtc score
    Parameters: 
    OA areas (df): The OAs areas in DSFRS and cas rates
    turnout times (df): The turnout times per hour per OA, either the base or the scenario
    drive times (df): The drive time per hour per output area for the first ten pumps
    scenario_name (str): name of the scenario or base model for the first run
    Returns: 
    dataframe of scenario name and scores
    """
    r = _create_final_turnout(turnout_times_df, drive_time_df, scenario_name)
    df = oa_areas_df.merge(r, how='left', on=('oa_code','hour'))
    df['dwelling_score'] = _calc_dwelling_fatalities(df['dwelling_cas_rate'].values, df['tt1'].values, df['tt2'].values)
    df['rtc_score'] = _calc_rtc_fatalities(df['rtc_cas_rate'].values, df['tt1'].values, df['tt2'].values)
    df = df.aggregate({"dwelling_score":['sum'],"rtc_score":['sum']}).reset_index(drop=True) 
    df['scenario'] = df.apply(lambda x: scenario_name if scenario_name != None else 'Base Case', axis = 1)
    df = df[['scenario','dwelling_score','rtc_score']]
    return df
def run_batch_scenarios(scenario_list = None):
    """ 
    Run the base model and any other scenario required
    Parameters: 
    scenario_list (list): The scenario name and actual scenario in a nested dictionary
    Returns: 
    dataframe of the dwel fire and rtc score for the base model plus scenario/s
    """
    scenario_count = len([s[0] for s in scenario_list])
    print(f'...................... Running modeller for {scenario_count} scenarios ......................')
    #o = _create_oa_areas()
    output_areas = pd.read_csv(dir_path + "/oa_master.csv")
    cas_rate = pd.read_csv(dir_path + "/base_cas_master.csv")
    o = pd.merge(output_areas, cas_rate,  how='left', left_on=['oa_code'], right_on = ['oa_code'])
    o.columns = ['oa_code', 'hour', 'dwelling_cas_rate', 'rtc_cas_rate']
    t = pd.read_csv(dir_path + "/turnout_time_master_orig.csv")
    drive_time = pd.read_csv(dir_path + "/drive_time_master.csv")
    hours = pd.DataFrame({'hour': list(range(0, 24)), 'key': 0})
    drive_time['key'] = 0
    d = drive_time.merge(hours, how='outer', on='key').drop(columns=['key'])
    b = _calculate_scores(o, t, d)
    print('Running base model')
    for scenario_name, scenario_dict in scenario_list:
        print(f'Running scenario: {scenario_name}')
        ts = pd.DataFrame.from_dict(_create_turnout_scenarios(scenario_dict), orient='index').reset_index(level=0).rename(columns={"index": "appliance_callsign"})
        ts = ts.melt(id_vars=['appliance_callsign'], var_name='hour', value_name='turnout_time')
        new_t = pd.merge(t, ts,  how='left', left_on=['appliance_callsign', 'hour'], right_on = ['appliance_callsign', 'hour'])
        new_t['turnout_time'] = new_t['turnout_time_y'].mask(pd.isnull, new_t['turnout_time_x'])
        new_t.drop(['turnout_time_x', 'turnout_time_y'], axis=1, inplace=True)
        s = _calculate_scores(o, new_t, d, scenario_name)
    return  pd.concat([b, s]).reset_index(drop=True)

# create batch scenarios to make pumps off
scenario_list = list()
scenario_list.append(
    ('Remove Crownhill', {
    "KV49P1": "off",
    "KV49P2": "off"
    }))

print(run_batch_scenarios(scenario_list))
