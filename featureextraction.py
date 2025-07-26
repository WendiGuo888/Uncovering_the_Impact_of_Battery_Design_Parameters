# -*- coding: utf-8 -*-
"""

@author: wengu476

This file extracts physically interpretable features from structured data.

"""

#%%
import h5py
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d
import statistics
import matplotlib.pyplot as plt
from scipy.io import loadmat
#%%
def process_data_mat(filename):
    data = []  # list for storing data
    
    
    try:        
        # Open .mat file
        with h5py.File(filename, 'r') as mat_file:
            
            # get the list of top-level field names that start with cycle
            file_name = list(mat_file.keys())[1]
            cycle_fields = [field for field in mat_file[file_name] if field.startswith('cycle')]
    
            # loop through each 'cycle' field
            for cycle_field in cycle_fields:
                cycle_data = mat_file[file_name][cycle_field]
    
                # check if it contains child nodes
                if 'CCCV_Chg' not in cycle_data or 'OCV' not in cycle_data or 'DChg' not in cycle_data:
                    continue  
    
                # iterate over the third-level fields ('CCCV_Chg', 'OCV', 'DChg')
                for subfield_name in ['CCCV_Chg', 'OCV', 'DChg']:
                    subfield_data = cycle_data[subfield_name]
    
                    # iterate over the fourth-level fields （'V', 'I', 'Q', 'T'）
                    for subsubfield in ['V', 'I', 'Q', 'T']:
                        subsubfield_values = subfield_data[subsubfield][:].tolist()
    
                        data_dict = {
                            'Cycle': cycle_field,
                            'Subfield': subfield_name,
                            'Subsubfield': subsubfield,
                            'Value': subsubfield_values
                        }
    
                        # add the dictionary to the list
                        data.append(data_dict)
    
                        # free memory
                        del subsubfield_values
                        
    except (OSError, KeyError) as e:
        print(f"h5py failed to open the file. Trying loadmat. Error: {e}")
        
        # if opening with h5py fails, use loadmat instead
        mat_data = loadmat(filename)

        # extract the required data based on the file structure
        if 'data_field' in mat_data:  # replace with the actual top-level field name
            top_level_data = mat_data['data_field']
            for cycle_field in top_level_data:
                if 'CCCV_Chg' in cycle_field and 'OCV' in cycle_field and 'DChg' in cycle_field:
                    for subfield_name in ['CCCV_Chg', 'OCV', 'DChg']:
                        subfield_data = cycle_field[subfield_name]
                        for subsubfield in ['V', 'I', 'Q', 'T']:
                            if subsubfield in subfield_data:
                                subsubfield_values = subfield_data[subsubfield].flatten().tolist()

                                # Create a dictionary containing data and field information
                                data_dict = {
                                    'Cycle': cycle_field,
                                    'Subfield': subfield_name,
                                    'Subsubfield': subsubfield,
                                    'Value': subsubfield_values
                                }

                                # append the dictionary to the list
                                data.append(data_dict)

                                # free memory
                                del subsubfield_values                    

    # create a dataframe
    df = pd.DataFrame(data)

    # create a multi-level dataframe using MultiIndex
    multi_index = pd.MultiIndex.from_frame(df[['Cycle', 'Subfield', 'Subsubfield']])
    df.set_index(multi_index, inplace=True)
    df.drop(columns=['Cycle', 'Subfield', 'Subsubfield'], inplace=True)

    return df,cycle_fields


def process_data_other(filename):
    try:
        # try to open the .mat file using h5py
        with h5py.File(filename, 'r') as input_file:
            input_fields = list(input_file.keys())
            output = np.array(input_file[input_fields[0]])
            return output
        
    except (OSError, KeyError) as e:
        print(f"h5py failed to open the file. Trying loadmat. Error: {e}")
        
        try:
            # if opening with h5py fails, fall back to using loadmat
            mat_data = loadmat(filename)
            
            # extract the calue of the first top-level key
            first_key = list(mat_data.keys())[3] 
            output = np.array(mat_data[first_key])
            return output
        
        except Exception as e:
            print(f"Failed to open file with loadmat: {e}")
            raise


def shannon_entropy(xdata,binnumber):
    # use the numpy.histogram function to calculate frequencies and bins
    # counts_temp, bin_edges_temp = np.histogram(xdata, bins='scott', density=True)
    # counts, bin_edges = np.histogram(xdata, bins=len(counts_temp)-1, density=True)
    counts, bin_edges = np.histogram(xdata, bins=binnumber, density=True)

    # replace zero frequencies with a small non-zero value
    counts[counts == 0] = 1e-10

    # calculate the probability of each individual value
    probabilities = counts / np.sum(counts)

    # calculate Shannon entropy
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))

    return shannon_entropy


def movmean(data, window_size):
    
    smoothed_data = []
    if window_size % 2 == 0:
        for i in range(len(data)):
            start = max(0, i - 1- (window_size-2) // 2 )
            end = min(len(data), i + (window_size - 2) // 2)
            window = data[start:end+1]
            smoothed_data.append(np.mean(window))
    else:
        for i in range(len(data)):
            start = max(0, i - (window_size-1) // 2 )
            end = min(len(data), i + (window_size - 1) // 2)
            window = data[start:end+1]
            smoothed_data.append(np.mean(window))
                

    return np.array(smoothed_data)

def calculate_slope(data):
    x = np.arange(len(data))
    return np.polyfit(x, data, 1)[0]

def features_extraction (datafile, EOCVfile, FECfile, RPT_FECfile, RPT_SOHfile, V_min, V_max, I_min, I_max):
        
    #  process the data file using a function and return a dataframe
    dataframe,cycle_eff = process_data_mat(datafile)
    EOCV = process_data_other(EOCVfile)
    FEC_final = process_data_other(FECfile)
    N_FEC_cumulative = process_data_other(RPT_FECfile)
    SOH_test_06CS1 = process_data_other(RPT_SOHfile)
    
    # extract features based on fragmentary data
    Numbercycles = np.size(EOCV) # total cycle numbers

    
    
    V_data = []
    stdValues_V = []
    V_fragment = []
    trmean_V = []
    min_V = []
    kurt_V = []
    skew_V = []
    shannonEntropy_V = []
    stdValues_V = []
    slope_V = []
    
    Q_data = []
    Q_fragment = []
    stdValues_Q = []
    shannonEntropy_Q = []
    trmean_Q = []  # store the median of Q
    min_Q = []    
    kurt_Q = []  
    skew_Q = []   
    stdValues_Q = []  
    slope_Q = [] 
    
    I_data = []
    I_fragment = []
    stdValues_I = []
    shannonEntropy_I = []
    trmean_I = []  # store the median of I
    kurt_I = []   
    skew_I = [] 
    stdValues_I = []  
    slope_I = [] 
    
    
    ws = 5
    
    # iterate over each branch
    for i in range(Numbercycles):
        
        # get the voltage data of the current cycle
        cycle_name = cycle_eff[i]
        
        # check if the data exists
        if (cycle_name, 'CCCV_Chg', 'V') in dataframe.index:
    
            V_selected = dataframe.loc[(cycle_name, 'CCCV_Chg', 'V')]
            Q_selected = dataframe.loc[(cycle_name, 'CCCV_Chg', 'Q')]
            I_selected = dataframe.loc[(cycle_name, 'CCCV_Chg', 'I')]

        
        I_data.append(I_selected.tolist()[0][0])
        
        # find the indices that meet the range condition
        I_data_array = np.array(I_data[i])
        
        I_indices = np.where((I_data_array >= I_min) & (I_data_array <= I_max))[0]
        
        I_fragment.append(I_data_array[I_indices])
        
        trmean_I.append(np.median(I_fragment[i]))
        kurt_I.append(kurtosis(I_fragment[i],fisher=False))
        skew_I.append(skew(I_fragment[i]))
        stdValues_I.append(np.std(I_fragment[i]))
        shannonEntropy_I.append(shannon_entropy(I_fragment[i],'scott'))
        slope_I.append(calculate_slope(I_fragment[i]))
                 
        
        V_data.append(V_selected.tolist()[0][0])
        
        # find the indices that meet the range condition
        V_data_array = np.array(V_data[i])
        
        indices = np.where((V_data_array >= V_min) & (V_data_array <= V_max))[0]
         
        V_fragment.append(V_data_array[indices])
        
        trmean_V.append(np.median(V_fragment[i]))
        min_V.append(np.min(V_fragment[i]))
        kurt_V.append(kurtosis(V_fragment[i],fisher=False))
        skew_V.append(skew(V_fragment[i]))
        stdValues_V.append(np.std(V_fragment[i]))
        shannonEntropy_V.append(shannon_entropy(V_fragment[i],'scott'))
        slope_V.append(calculate_slope(V_fragment[i]))
    
        # extract Q segment
        Q_data.append(Q_selected.tolist()[0][0])
        Q_data_array = np.array(Q_data[i])
        Q_fragment.append(Q_data_array[indices])
    
        # calculate the standard deviation and Shannon entropy
        trmean_Q.append(np.median(Q_fragment[i]))
        min_Q.append(np.min(Q_fragment[i]))
        kurt_Q.append(kurtosis(Q_fragment[i], fisher=False))
        skew_Q.append(skew(Q_fragment[i]))
        stdValues_Q.append(np.std(Q_fragment[i]))  
        shannonEntropy_Q.append(shannon_entropy(Q_fragment[i],len(Q_fragment[i])))
        slope_Q.append(calculate_slope(Q_fragment[i]))
    
    # move average the feature trend    
    trmean_V = movmean(trmean_V, ws)
    min_V = movmean(min_V, ws)
    kurt_V = movmean(kurt_V, ws)
    skew_V = movmean(skew_V, ws)
    stdValues_V = movmean(stdValues_V, ws)
    shannonEntropy_V = movmean(shannonEntropy_V, ws)
    slope_V = movmean(slope_V, ws)
    
    trmean_Q = movmean(trmean_Q, ws)
    min_Q = movmean(min_Q, ws)
    kurt_Q = movmean(kurt_Q, ws)
    skew_Q = movmean(skew_Q, ws)
    stdValues_Q = movmean(stdValues_Q, ws)
    shannonEntropy_Q = movmean(shannonEntropy_Q, ws)
    slope_Q = movmean(slope_Q, ws)
    
    trmean_I = movmean(trmean_I, ws)
    kurt_I = movmean(kurt_I, ws)
    skew_I = movmean(skew_I, ws)
    stdValues_I = movmean(stdValues_I, ws)
    shannonEntropy_I = movmean(shannonEntropy_I, ws)
    slope_I = movmean(slope_I, ws)
      
    # ===extract features based on DQ===
    
    # find the largest dimension
    max_length=1000
    # iterate over all cells
    for i in range(Numbercycles):
        # get the length of the list in the current cell
        current_length = len(Q_fragment[i])
        
        # update the maximum dimension
        max_length = max(max_length, current_length)
    
       
    # initialize a list to store features
    V_fragment_interp = []
    Q_fragment_interp = []
    DQ_fragment = []
    
    trmean_DQ = []
    min_DQ = []     
    kurt_DQ = []   
    skew_DQ = [] 
    stdValues_DQ = []
    shannonEntropy_DQ = []
    slope_DQ = []
    
    # interpolate V and Q for each cell
    for i in range(Numbercycles):
        V = V_fragment[i]
        Q = Q_fragment[i]
    
        # compute the interpolated V and Q for the current cell
        V_interp = np.linspace(V_min, V_max, max_length)
    
        # remove duplicate V values and their corresponding Q values
        unique_indices = np.unique(V, return_index=True)[1]
        V_unique = V[unique_indices]
        Q_unique = Q[unique_indices]
    
        interp_func = interp1d(V_unique, Q_unique, kind='linear', fill_value='extrapolate')
        Q_interp = interp_func(V_interp)
      
        # assign the interpolated V and Q to the current cell
        V_fragment_interp.append(V_interp)
        Q_fragment_interp.append(Q_interp)
    
        # DQ std & shannon
        DQ = Q_interp - Q_fragment_interp[0]
        trmean_DQ.append(np.median(DQ))
        min_DQ.append(np.min(DQ))
        kurt_DQ.append(kurtosis(DQ, fisher=False))
        skew_DQ.append(skew(DQ))
        DQ_fragment.append(DQ)
        stdValues_DQ.append(np.std(DQ))
        shannonEntropy_DQ.append(shannon_entropy(DQ ** 2,'scott'))
        slope_DQ.append(calculate_slope(DQ_fragment[i]))
        
    trmean_DQ = movmean(trmean_DQ, ws)
    min_DQ = movmean(min_DQ, ws)
    kurt_DQ = [3 if np.isnan(m) else m for m in kurt_DQ]
    skew_DQ = [3 if np.isnan(z) else z for z in skew_DQ]
    kurt_DQ = movmean(kurt_DQ, ws)
    skew_DQ = movmean(skew_DQ, ws)
    stdValues_DQ = movmean(stdValues_DQ, ws)
    shannonEntropy_DQ = movmean(shannonEntropy_DQ, ws)
    slope_DQ = movmean(slope_DQ, ws)
        

    
    # define the data segments based on charging/discharging behavior
    V_increasing = []
    V_constant = []
    Q_increasing = []
    Q_constant = []
    
    sequence_CC = []
    sequence_CV = []
    
    # define the data segments based on charging/discharging behavior       
    for i in range(len(V_data)):
        V = V_data[i]
        Q = Q_data[i]
        
        # 过滤掉小于3.5V的电压值及其对应的电量值
        filtered_indices = np.where(np.array(V) >= V_min)[0]
        V = [V[index] for index in filtered_indices]
        Q = [Q[index] for index in filtered_indices]
        
        # 如果过滤后没有足够的数据，则跳过当前循环
        if not V:
            continue
        
        V_arrry = np.array(V)
        # 找到众数
        mode_value = statistics.mode(V_arrry)
        
        # 找到不变序列的开始索引
        start_index = np.argmax(V_arrry >= mode_value)
        
        # 将递增和不变序列存储为新的变量
        V_increasing.append(V[:start_index])  # CC
        V_constant.append(V[start_index:])    # CV
        Q_increasing.append(Q[:start_index])  # CC
        Q_constant.append(Q[start_index:])    # CV
        
        sequence_CC.append(len(V_increasing[i]))
        sequence_CV.append(len(V_constant[i]))
        
    
    # ===Extract features based on IC===
    
    smooth_IC = []
    interpolated_V = []
    interpolated_Q = []
    V_peak = []
    IC_peak = []
    IC_area = []
    
    for i in range(Numbercycles):
        # 将数据存储为矩阵
        X = np.array(V_increasing[i])
        Y = np.array(Q_increasing[i]) * 1e-3  
 
        # perform deduplication to retain only unique sampling points
        unique_idx = np.unique(X, return_index=True)[1]
        unique_V = X[unique_idx]
        unique_Q = Y[unique_idx]
        
        # find the third largest value and use it as xmax
        sorted_idx = np.argsort(unique_V)[::-1]
        largest_X = unique_V[sorted_idx[0]]
        
        # set xmax
        xmax = largest_X
        
        # set the number of interpolation points
        m = 500
        
        # generate m evenly spaced interpolation points between xmin and xmax
        interpolated_V.append(np.linspace(min(unique_V), xmax, m))
        
        # obtain m Y values through linear interpolation
        interpolated_Q.append(np.interp(interpolated_V[i], unique_V, unique_Q))
        
        # compute the first derivative of the interpolated X
        dx = interpolated_V[i][1] - interpolated_V[i][0]
        IC = np.diff(interpolated_Q[i]) / dx
        
        # smooth the derivative using moving average filtering
        window_size = 10  # moving average window size
        smooth_IC.append(movmean(IC,window_size))
        
        # get the maximum value of the IC vector and its corresponding index
        max_index = np.argmax(smooth_IC[i])
        
        # store the index of the peak value
        IC_peak.append(np.max(smooth_IC[i]))
        
        # get the V value at the corresponding index (position value)
        V_peak.append(interpolated_V[i][max_index + 1])
        
        # filter the data for the specified voltage range
        truncated_voltage = interpolated_V[i][:-1]
    
        # generate a mask for smooth_IC[i]
        mask = (truncated_voltage >= V_min) & (truncated_voltage <= V_max)
        filtered_voltage = truncated_voltage[mask]
        filtered_IC = smooth_IC[i][mask]
        
        # check if there are enough points to calculate the area
        if len(filtered_voltage) < 2:
            IC_area.append(0)
        else:
            # calculate the area using the trapezoidal rule
            area = np.trapz(filtered_IC, filtered_voltage)
            IC_area.append(area)
        

        
    #===Extract features based on DV===
    
    # 初始化存储数据的列表
    interpolated_Q_DV = []
    interpolated_V_DV = []
    smooth_DV = []
    last_index = []
    Q_DV = []
    DV_Valley = []
    
    # 创建图形窗口
    plt.figure()
    
    for i in range(Numbercycles):
        # 提取数据
        Y = np.array(V_increasing[i])
        X = np.array(Q_increasing[i]) * 1e-3
        
        # perform deduplication to retain only unique sampling points
        unique_idx = np.unique(X, return_index=True)[1]
        unique_Q = X[unique_idx]
        unique_V = Y[unique_idx]
        
    
        # find the third largest value and use it as xmax
        sorted_idx = np.argsort(unique_Q)[::-1]
        largest_X = unique_Q[sorted_idx[0]]
    
        # set xmax
        xmax = largest_X
    
        # set the number of interpolation points
        m = 500
    
        # generate m equally spaced interpolation points between xmin and xmax
        interpolated_Q_DV.append(np.linspace(min(unique_Q), xmax, m))
    
        # obtain m Y values through linear interpolation
        f = interp1d(unique_Q, unique_V, kind='linear')
        interpolated_V_DV.append(f(interpolated_Q_DV[i]))
    
        # calculate DV
        dx = interpolated_Q_DV[i][1] - interpolated_Q_DV[i][0]
        DV = np.diff(interpolated_V_DV[i]) / dx
    
        #  Smooth the derivative using moving average filtering   
        window_size = 10  # moving average window size
        smooth_DV.append(movmean(DV,window_size))
        DV_Valley.append(np.min(smooth_DV[i]))
    
        # get the last index of the vector
        last_index = np.size(smooth_DV[i])
    
        # get the Q value at the corresponding index (position value)
        Q_DV.append(interpolated_Q_DV[i][last_index])


    # ===Generate the features matrix===
    
    # perform linear interpolation using interp1d
    interp_func =  interp1d(N_FEC_cumulative.T[0], SOH_test_06CS1[0], kind='linear', fill_value='extrapolate')
    
    Qloss_RPT = 1 - SOH_test_06CS1
    
    # compute the interpolated SOH (SOH_interp)
    SOH_interp = interp_func(FEC_final.T[0])
    
    # define the SOH threshold for end-of-life (EOL) of the battery
    EOL_SOH = 0.8

    # find the first point where SOH drops below the threshold
    eol_indices = np.where(SOH_interp <= EOL_SOH)[0]
    if len(eol_indices) > 0:
        eol_index = eol_indices[0]  # the first index that meets the condition
    else:
        eol_index = len(SOH_interp) - 1  # if no matching condition is found, use the last index
    
    eol_cycle = FEC_final[eol_index]  # cycle number at the expected end of life

    # calculate the RUL for each cycle
    RUL = eol_cycle - FEC_final  # RUL sequence
    
    Inputdata = np.column_stack((Q_DV, IC_peak, V_peak, IC_area,
                                trmean_I, kurt_I, skew_I, stdValues_I, shannonEntropy_I, slope_I,
                                trmean_V, skew_V, stdValues_V, shannonEntropy_V, slope_V,
                                kurt_Q, skew_Q, stdValues_Q, 
                                trmean_DQ, min_DQ, kurt_DQ, skew_DQ, stdValues_DQ, shannonEntropy_DQ, slope_DQ,
                                sequence_CC, sequence_CV, FEC_final, EOCV[0], SOH_interp, RUL))

    
    
    column_names = ['Q_DV', 'IC_peak', 'V_peak', 'IC_area', 
                    'media_I', 'kurt_I', 'skew_I', 'std_I','shanEntro_I', 'slope_I',
                    'media_V', 'skew_V', 'std_V','shanEntro_V', 'slope_V',
                    'kurt_Q', 'skew_Q', 'std_Q', 
                    'media_DQ', 'min_DQ', 'kurt_DQ', 'skew_DQ', 'std_DQ','shanEntro_DQ','slope_DQ',
                    'seq_CC','seq_CV', 'EFC', 'EOCV', 'SOH', 'RUL']
    

    
        
    
    # convert the NumPy array to a Pandas DataFrame and assign column names
    df = pd.DataFrame(Inputdata, columns=column_names)
    

    print('The features are listed in the order of', column_names)
    
           
    return df, N_FEC_cumulative, Qloss_RPT, SOH_interp, FEC_final, RUL #feature df, RPT EFC, RPT Qloss, Qloss interp, FEC interp, RUL

#%%
if  __name__ == '__main__':
    features1, FECs1, Qlosss1, RPTs1, FEC_finals1, RULs1 = features_extraction('2C0degCs1/data.mat', '2C0degCs1/EOCV.mat', '2C0degCs1/EFC.mat', '2C0degCs1/RPT_EFC.mat', '2C0degCs1/SOH.mat', V_min=3.5, V_max=4.0, I_min=100, I_max=500)
