import math
import numpy as np
from scipy.stats import rv_continuous
# epsilon：ε，总体隐私预算
# epsilon_reminded：。。。前面的epsilon_remained，就是ε'
# sample_result: 采样点集sample_points
# score_array:会有0的风险
class CustomDistribution(rv_continuous):
    #x就是那个v*
    def _pdf(self, x, v, q, b, epsilon):
        d = np.abs(x-v)
        return (q * np.exp(-epsilon*d))

    def _argcheck(self, v, q, b, epsilon):
        result = np.isfinite(v)
        return result


def perturb(data, n, alpha, w, theta, mu,  score_array, epsilon, epsilon_reminded, sample_result):

        # equation11中的比例公式，其中equation11中的β=1-α
        # 这里的p值计算中，会有可能出现/0错误，因为scorearray有问题: 有=0或<0的风险
        if score_array[n]==0.0 or score_array[n]<0:
            score_array[n] = 0.00001
        p = round(1 - np.exp(-1 * (alpha / score_array[n] + (1 - alpha) * score_array[n])), 5)



        # p = 1 - math.exp(-score_array[n])
        # b: 自适应误差边界，会根据采样点的重要性更新，即根据score_array进行更新，重要的点，b比较小，浮动较小
        # epsilon_now: 给当前timestamp分配的隐私预算
        # epsilon_reminded: ε',分配完之后剩多少ε了
        # q: q概率报告真实v，1-q概率报告扰动v'，equation14
        # perturbed_result: v',扰动后的输出值


        b = round(np.log(theta/score_array[n]+mu), 5)
        epsilon_now = p*epsilon_reminded
        epsilon_reminded -= epsilon_now
       # q的计算容易溢出，try，except处理一下

        q = round(0.5 * epsilon_now / (1 - np.exp(-epsilon_now * b)), 5)
        if math.isnan(q):
            q = round(0.5*epsilon_now, 5)
        #原来的
        # perturb_value = sample_function(q, b, epsilon_now) #这两个没看懂，pertub_value得到一个扰动范围？，pertubed_value则是最终输出的扰动值
        # perturbedV = data[sample_result[n]]+perturb_value
        # print("alpha", alpha)
        # perturbed_result = sample_function2(data[sample_result[n]], q, b, epsilon)
        #我的
        # print("正在输出sampleresult[n]",sample_result[n],"data 为：",data[sample_result[n]],"scoreArray:",score_array[n],"b:",b)
        perturbedV = sample_function2(data[sample_result[n]], q, b, epsilon)

        # print("原值为", data[sample_result[n]], "，扰动后为：",perturbedV)

        # 对控制p的权重系数α的动态变化，这里变完之后直接返回变完后的α给下一个使用就可以了
        if epsilon_reminded>epsilon/2:
            alpha -= 1/((1-alpha)*(1-alpha))*0.01
        elif epsilon/w > epsilon_reminded:
            alpha += 1/((1-alpha)*(1-alpha))*0.01
        return alpha, perturbedV, epsilon_reminded


# 扰动方法？这个方法应该是经过奇怪的逻辑改造过？
# 感觉有点问题，因为隐私预算增加，按理来说同个数据集DTW应该呈降低态势才对。。
def sample_function(q, b, epsilon):
    number = np.random.random_sample() #生成一个[0,1)之间的随机数
    if number < 0.5:
        result = np.log(number*q/epsilon+np.exp(-1*epsilon*b))/epsilon
        return result
    else:
        result = -1*np.log((0.5+q/epsilon-number)*epsilon/q)/epsilon
        return result



def sample_function2(v, q, b, epsilon):
    root_number = np.random.random_sample();
    if root_number <= q:
        result = v
        return result
    else: #从输出集中选一个答案输出
        output_range_down = v-b
        output_range_up = v+b

        distribution = CustomDistribution(a=output_range_down, b=output_range_up, name="PrDistribution")
        # xlist = np.arange(output_range_down, output_range_up, 1)
        # ylist = distribution.pdf(xlist, v, q, b, epsilon)
        # samples = distribution.rvs(xlist, v, q, b, epsilon)
        try:
            samples = distribution.rvs(v, q, b, epsilon)
        except RuntimeError:
            print("发生了RuntimeError：v:",v,",q:",q,",b:",b,",epsilon:",epsilon)
        # print("down:",output_range_down,", up:",output_range_up,"abs:",np.abs(int(output_range_up - output_range_down)+1)  ,"samples参数:",xlist, v, q, b, epsilon,"samples：",samples)

        # distribution = CustomDistribution(a=output_range_down, b=output_range_up, name="PrDistribution")
        # xlist = np.arange(output_range_down, output_range_up, 1)
        # ylist = distribution.pdf(xlist, v, q, b, epsilon)
        #
        # samples = distribution.rvs(v, q, b, epsilon, size=200)

        if type(samples) == np.ndarray:
            result = samples[0]

        elif isinstance(samples, float):
            result = samples

        # print("result的type",type(res))
        return result





