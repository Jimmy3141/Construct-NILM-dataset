# sample_result: 就是采样点集，前面的sample_points
# n：从第3个开始数的采样点
# 返回：

def pid_control(n, data, time, error_array, sample_result, k_p, k_i, k_d, pi, score_array):
    # k：n的前2个采样点之间的斜率
    # error_array: equation7的内容，存储每个采样点的|真实值-预测值|。其中，预测值用 【k*（差值t）+上一个v】来表示
    k = (data[sample_result[n-1]]-data[sample_result[n-2]])/(time[sample_result[n-1]]-time[sample_result[n-2]])
    error_array.append(abs(k*(time[sample_result[n]]-time[sample_result[n-1]])+data[sample_result[n-1]]
                           - data[sample_result[n]]))
    score = k_p*error_array[n]

    if n > 0:
        if n-pi-1 >= 0: # 这个ifelse有什么需要吗。。。
            for i in range(n-pi-1, n+1):
                score += k_i/pi*error_array[i]  # equation8
        else:
            for i in range(0, n + 1):
                score += k_i / pi * error_array[i]
        score += k_d*(error_array[n]-error_array[n-1])/(time[sample_result[n]]-time[sample_result[n-1]]) # equation8

    score_array.append(score)

    # score_array: 存储每个采样点的importance，即equation8
    return score_array, error_array
