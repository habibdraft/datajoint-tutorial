import pandas as pd
import datajoint as dj
import numpy as np

mouse = pd.read_pickle('ret1_data.pkl')
df = pd.DataFrame(mouse)

df = pd.concat([df.drop(['stimulations'], axis=1), df['stimulations'].apply(pd.Series)], axis=1)

df1 = df.drop(0, 1)
df2 = df.drop(1, 1)
df2.rename(columns={0:1}, inplace=True)
df1 = pd.concat([df1.drop([1], axis=1), df1[1].apply(pd.Series)], axis=1)
df2 = pd.concat([df2.drop([1], axis=1), df2[1].apply(pd.Series)], axis=1)
df = pd.concat([df1, df2], axis=0).reset_index().drop(0, 1).drop('index', 1)
df['session_id'] = df.index
df['stim_duration'] = 1/df['fps']
df = df[df['spikes'].notnull()]

def get_stim_times(onset, duration, frames):
    time_since_onset = onset
    stim_times = []
    for stim in range(int(frames)):
        time_since_onset += duration
        stim_times.append(time_since_onset)
    stim_times = np.array(stim_times)
    return stim_times

df['stim_times'] = df.apply(lambda x: get_stim_times(x['stimulus_onset'], x['stim_duration'], x['n_frames']), axis=1)

def generate_spike_triggered_array(spike_times, stim_times):
    timesteps = 150
    stim_times *= 1000
    sta_list = []
    for neuron in spike_times:
        sta = np.zeros((timesteps,))
        neuron *= 1000
        spike_indices = neuron[timesteps:].nonzero()[0] + timesteps
        for spike_index in spike_indices:
            sta += stim_times[spike_index-timesteps:spike_index]
            sta /= len(neuron)
            sta_list.append(sta)
    return(sta_list)

df['spike_triggered_array'] = df.apply(lambda x: generate_spike_triggered_array(x['spikes'], x['stim_times']), axis=1)

session_records = df.to_dict(orient='records')

dj.config["enable_python_native_blobs"] = True
schema = dj.schema('nhabib_tutorial', locals())  

@schema
class Session(dj.Manual):
      definition = """
      session_id: int                  
      ---
      sample_number: int                      
      spikes: longblob
      stim_width = NULL : int
      stimulus_onset = NULL : float
      fps = NULL : float
      pixel_size = NULL : float
      y_block_size = NULL : int
      stim_height = NULL : int
      n_frames = NULL : int
      x_block_size = NULL : int
      movie: longblob
      subject_name = NULL : varchar(100)
      session_date: date
      stim_duration = NULL : float
      stim_times: longblob
      spike_triggered_array: longblob
      """
 session = Session()
 session.insert(session_records)
