import os
import csv
import json
import shutil
import numpy as np
import pandas as pd
import scipy.special
import imblearn.over_sampling
from os import listdir
from os.path import isfile, join
from itertools import combinations
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


# def processInstance(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate):
#
#     attrs = ["A", "B", "C", "D"] #selected combination of input attributes ... n
#     # ...extend or delete depending on the attributes you choose to evaluate
#
#     balancingTypes = ['SMOTE','RANDOMOVERSAMPLING', 'RANDOMUNDERSAMPLING']
#
#     for b in range(1,len(balancingTypes)):
#         balancingType = balancingTypes[b]
#
#         for r in range(1,len(attrs)):
#             attcombinations = list(combinations(attrs, r))
#
#             for i in range(len(attcombinations)):
#                 input_nodes = len(attcombinations[0])
#                 attcombination = attcombinations[i]
#
#                 configuration = {
#                     "InputNodes":input_nodes,
#                     "HiddenNodes":hidden_nodes,
#                     "OutputNodes":output_nodes,
#                     "epochs":epochs,
#                     "LearningRate":learning_rate,
#                     "BalancingType":balancingType,
#                     "attcombination":attcombination
#                 }
#
#                 configurationExist = Utilities.configexist(configuration)
#                 if (configurationExist == None):
#                     annf = NeuralNetworkFactory(configuration)
#                     configuration['performance'] = annf.execute(configuration)
#                     Utilities.writeConfigPerformance(configuration)
#                     # Utilities.prettyprintconfig(configuration,True)
#                 else:
#                     configuration = configurationExist
#                     # Utilities.prettyprintconfig(configuration,False)
#     pass

def processInstance(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate):
    attrs = ["A", "B", "C", "D"]  # Selected combination of input attributes
    balancingTypes = ['SMOTE', 'RANDOMOVERSAMPLING', 'RANDOMUNDERSAMPLING']

    for b in range(len(balancingTypes)):  # 从0开始遍历所有平衡方法
        balancingType = balancingTypes[b]

        for r in range(1, len(attrs)):  # 从1开始，因为我们希望至少选择一个属性
            attcombinations = list(combinations(attrs, r))

            for i in range(len(attcombinations)):
                input_nodes = len(attcombinations[i])  # 确保正确获取当前组合的输入节点数
                attcombination = attcombinations[i]

                configuration = {
                    "InputNodes": input_nodes,
                    "HiddenNodes": hidden_nodes,
                    "OutputNodes": output_nodes,
                    "epochs": epochs,
                    "LearningRate": learning_rate,
                    "BalancingType": balancingType,
                    "attcombination": attcombination
                }

                configurationExist = Utilities.configexist(configuration)
                if configurationExist is None:
                    annf = NeuralNetworkFactory(configuration)
                    performance = annf.execute(configuration)
                    if performance:
                        configuration['performance'] = performance
                        Utilities.writeConfigPerformance(configuration)
                        # 在这里将结果写入 CSV 文件
                        write_result_to_csv(configuration, performance)
                else:
                    print("Configuration already exists. Skipping...")

# 写入结果到 CSV 文件的函数
def write_result_to_csv(configuration, performance):
    results_file = f"data/_results{configuration['epochs']}/results.csv"
    with open(results_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        row = [
            configuration["InputNodes"], configuration["LearningRate"], configuration["HiddenNodes"],
            configuration["OutputNodes"], configuration["epochs"],
            "N/A",  # PreBalance:ClassDistributionRatio 可以根据需要填写
            configuration["BalancingType"],
            "N/A",  # PostBalance:ClassDistributionRatio 可以根据需要填写
            "N/A",  # truepositiverate
            "N/A",  # falsepositiverate
            "N/A",  # precision
            "N/A",  # recall
            performance.get("accuracy", "N/A"),
            "N/A"  # f1score
        ]
        csv_writer.writerow(row)

class NeuralNetworkFactory:

    def __init__(self, seedconfig):
        self.inputNodes = seedconfig["InputNodes"]
        self.hiddenNodes = seedconfig["HiddenNodes"]
        self.outputNodes = seedconfig['OutputNodes']
        self.learningRate = seedconfig['LearningRate']
        self.epochs = seedconfig['epochs']
        self.attcombination = seedconfig['attcombination']
        self.balancingType = seedconfig['BalancingType']

    def getInputOutput(self, row_):

        # split the record by the ',' commas
        row = row_.split(',')

        if(row[len(row)-1] == '\n'):
            return None

        outputGroundTruth = float(row[len(row)-1])
        negative = 0.01
        positive = 0.01
        if(outputGroundTruth == 0):
            negative = 0.99
        elif(outputGroundTruth ==1):
            positive = 0.99
        else:
            return None
        
        inputs = []
        has_a = False
        has_b = False
        has_c = False
        has_d = False

        a = -1
        b = -1
        c = -1
        d = -1


        if("A" in self.attcombination):
            has_a = True
            a = float(row[1]) # 
            inputs.append(a)
        if("B" in self.attcombination):
            has_b = True
            if(has_a):
                b = float(row[2]) # 
            else:
                b = float(row[1]) # 
            inputs.append(b)
    
        if("C" in self.attcombination):
            has_c = True
            if(has_a and has_b):
                c = float(row[3]) # 
            elif(has_a and not has_b):
                c = float(row[2]) # 
            elif(not has_a and has_b):
                c = float(row[2]) # 
            elif(not has_a and not has_b):
                c = float(row[1]) # 
            
            inputs.append(c)

        if("D" in self.attcombination):
            has_d = True
            if(has_a and has_b and has_c):
                d = float(row[4]) # 
            elif(has_a and has_b and not has_c):
                d = float(row[3]) # 
            elif(has_a and not has_b and has_c):
                d = float(row[3]) # 
            elif(not has_a and has_b and has_c):
                d = float(row[3]) # 
            elif(has_a and not has_b and not has_c):
                d = float(row[2]) # 
            elif(not has_a and not has_b and has_c):
                d = float(row[2]) # 
            elif(not has_a and has_b and not has_c):
                d = float(row[2]) # 
            elif(not has_a and not has_b and not has_c):
                d = float(row[1]) # 

            inputs.append(d)

        # create target output value
        targets = [negative,positive]
        
        return [inputs, targets]

    def execute(self,configuration):

        self.balancingType = configuration['BalancingType']

        self.dataProcessor = DataProcessor(self.epochs,configuration, self.balancingType,self.attcombination)
        self.neuralNetwork = NeuralNetwork(self.inputNodes,self.hiddenNodes,self.outputNodes, self.learningRate)

        self.trainingDataList = self.dataProcessor.getTrainingData()
        # self.trainingDataList = self.dataProcessor.getCleanedTrainingData()
        
        attmetrics= []
        if("A" in self.attcombination):
            attmetrics.append('A') #            
        if("B" in self.attcombination):
            attmetrics.append('B') #            
        if("C" in self.attcombination):
            attmetrics.append('C') #            
        if("D" in self.attcombination):
            attmetrics.append('D') #
        
        attmetrics_str = ','.join(str(x) for x in attmetrics)
    
        for epoch in range(self.epochs):
            rowcount = 0

            #go through all the rows in the training data set
            for row_ in self.trainingDataList:
                rowcount = rowcount +1

                result = self.getInputOutput(row_)
                if(result == None):
                    continue
                inputs = result[0]
                targets = result[1]

                self.neuralNetwork.train(inputs, targets)

                rate = str(round((rowcount/len(self.trainingDataList))*100,2))+"%"
                # info = 'Training - epoch:'+str(epoch+1)+', Att:['+attmetrics_str+'], BT:'+str(self.balancingType)+', Row:'+rate
                info = 'Training - epoch:'+str(epoch+1)+', Att:['+attmetrics_str+'], BT:'+str(self.balancingType)+', Row:'+rate
                print(info, end="\r", flush=True)
                pass
            pass

        # test the neural network
        self.testingDataList = self.dataProcessor.getTestingData()

        # scorecard for how well the network performs, initially empty
        scorecard = []

        truepositive = 0
        truenegative = 0
        falsepositive = 0
        falsenegative = 0
        precision = 0
        recall = 0
        truepositiverate = 0
        falsepositiverate = 0
        sensitivity = 0
        specificity =0
        accuracy = 0
        f1score = 0

        rowcount = 0

         #go through all the rows in the testing data set
        for row_ in self.testingDataList:
            rowcount = rowcount +1

            result = self.getInputOutput(row_)
            if(result == None):
                continue

            # query the network
            inputs = result[0]
            correctOutput = result[1]
            predictedOutput = self.neuralNetwork.query(inputs)

            matched = False
            if(round(correctOutput[0]) == round(predictedOutput[0][0])):
                if(round(correctOutput[1]) == round(predictedOutput[1][0])):
                    # network's answer matches correct answer, add 1 to scorecard
                    scorecard.append(1)
                    matched = True

            if not (matched):
                scorecard.append(0)

            if(round(correctOutput[0]) == round(predictedOutput[0][0])):
                truenegative = truenegative +1
            else:
                falsenegative = falsenegative +1
            
            if(round(correctOutput[1]) == round(predictedOutput[1][0])):
                truepositive = truepositive +1
            else:
                falsepositive = falsepositive +1

            if((truepositive +falsepositive)>0):
                precision = truepositive/(truepositive +falsepositive)
            if((truepositive +falsenegative)>0):
                recall = truepositive/(truepositive +falsenegative)
            if((truepositive + falsenegative)>0):
                truepositiverate = truepositive/(truepositive + falsenegative)
            if((falsepositive + truenegative)>0):
                falsepositiverate = falsepositive/(falsepositive + truenegative)
            if((truepositive + falsenegative)>0):
                sensitivity = truepositive/(truepositive + falsenegative)
            if((truenegative + falsepositive)>0):
                specificity = truenegative/(truenegative + falsepositive)
            if((precision +recall)>0):
                f1score = 2 *((precision*recall)/(precision +recall))

            # calculate the performance score, the fraction of correct answers
            scorecard_array = np.asarray(scorecard)
            accuracy = scorecard_array.sum() / scorecard_array.size

            rate = str(round((rowcount/len(self.testingDataList))*100,2))+"%"
            f1score_round = str(round(f1score,4))
            
            info = 'Testing - Data:[epochs:'+str(self.epochs)+', Atts:['+attmetrics_str+'] BT:'+self.balancingType+', Row:'+rate+', F1-Score:'+f1score_round
            print(info, end="\r", flush=True)
            pass

        pass

        scorecard_array = np.asarray(scorecard)
        accuracy = scorecard_array.sum() / scorecard_array.size

        performance ={
            "accuracy":accuracy,
            "truepositive":truepositive,
            "truenegative":truenegative,
            "falsepositive":falsepositive,
            "falsenegative":falsenegative,
            "precision":precision,
            "recall":recall,
            "truepositiverate":truepositiverate,
            "falsepositiverate":falsepositiverate,
            "sensitivity":sensitivity,
            "specificity":specificity,
            "f1score":f1score,
            "accuracy":accuracy
            # "scorecard": scorecard
        }
        # print (json.dumps(performance))

        performance['scorecard']= scorecard
        return performance


class DataProcessor:
    def __init__(self,epoch, configuration_, _balancing_type, attcombination):
        util = Utilities()

        # mergeTrainingfiles = []
        # mergeTestingfiles = []

        self.balanced_data_file = "data/_balanced"+str(epoch)+"/balanced.csv"
        self.headless_file = "data/_headless"+str(epoch)+"/headerless.csv"
        self.training_data_file = "data/_training"+str(epoch)+"/train.csv"
        self.testing_data_file = "data/_testing"+str(epoch)+"/test.csv"

        #balance data (NB: Replace _sample.csv with your ground truth data from repocrawler)
        util.balanceData(epoch,configuration_,"grounddata/_sample.csv",self.balanced_data_file, _balancing_type,attcombination)

        #load training data for aws
        util.removeHeader(self.balanced_data_file,self.headless_file)
        util.splitTrainTestData(self.headless_file,self.training_data_file,self.testing_data_file)

    def getTrainingData(self):
        training_data_ = open(self.training_data_file, 'r')
        training_data_list = training_data_.readlines()
        training_data_.close()

        return training_data_list

    def getCleanedTrainingData(self):
        self.cleanTrainingData()

        training_data_ = open(self.training_clean_data_file, 'r')
        training_data_list = training_data_.readlines()
        training_data_.close()

        return training_data_list
    
    def getTestingData(self):
        testing_data_ = open(self.testing_data_file, 'r')
        testing_data_list = testing_data_.readlines()
        testing_data_.close()

        return testing_data_list

    
#Neural network class definition
class NeuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set the number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who. 
        # Weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
       
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass


    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    

# 


class Utilities:
        
    #balance dataset
    def balanceData(self,epoch,configuration,f1_unbalanced,f2_balanced, _balancing_type,attcombination):
        #check the class frequency using value_counts and find the class distribution ratio.
        # print(f1_unbalanced)
        data_unbalanced = pd.read_csv(f1_unbalanced)
        data_unbalanced['Vulnerability_Truth'].value_counts()
        positives = len(data_unbalanced[data_unbalanced['Vulnerability_Truth'] == 1])
        negatives = len(data_unbalanced[data_unbalanced['Vulnerability_Truth'] == 0])

        class_distribution_ratio = negatives/positives
        info = 'class distribution ratio (Vulnerability vs NonVulnerability)- raw data:'+str(positives)+'/'+str(negatives)
        
        prebalance = {
            "ClassDistributionRatio":class_distribution_ratio,
            "NoVulnerability":positives,
            "NoNonVulnerability":negatives
        }

        configuration['PreBalance'] = prebalance

        rows = []
        # header = ["A", "B", "C", "D", ...,"Vulnerability_Truth"]
        header = []
        if("A" in attcombination):
            header.append("A") # 
        if("B" in attcombination):
            header.append("B") # 
        if("C" in attcombination):
            header.append("C") # 
        if("D" in attcombination):
            header.append("D") #         
        # ...extend or delete depending on the attributes you choose to evaluate
        
        header.append("Vulnerability_Truth")

        try:
            with open(f1_unbalanced) as file_obj: 
                reader_obj = csv.DictReader(file_obj)
                
                for row in reader_obj:  
                    vulnerabilityTruth = row["Vulnerability_Truth"]               

                    if(vulnerabilityTruth == "0"):
                        pass
                    elif(vulnerabilityTruth == "1"):
                        pass
                    else:
                        continue
                        
                    a = 0
                    b = 0   
                    c = 0
                    d = 0
                    # ...extend or delete depending on the attributes you choose to evaluate
                
                    a = float(row["A"])  # 
                    b = float(row["B"])  # 
                    c = float(row["C"])  # 
                    d = float(row["D"])  # 
                    # ...extend or delete depending on the attributes you choose to evaluate

                    t_input = []
                    if("A" in attcombination):
                        t_input.append(a)
                    if("B" in attcombination):
                        t_input.append(b)
                    if("C" in attcombination):
                        t_input.append(c)
                    if("D" in attcombination):
                        t_input.append(d)
                    # ...extend or delete depending on the attributes you choose to evaluate                        

                    t_input.append(vulnerabilityTruth)

                    rows.append(t_input)                
        except Exception as e:
            print(f"Error: {e}")

        pass        

        with open("data/_attended"+str(epoch)+"/data_self_attended.csv", 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            csv_writer.writerows(rows)
            
        data_self_attended = pd.read_csv("data/_attended"+str(epoch)+"/data_self_attended.csv")

        data_balanced = []
        if(_balancing_type == 'SMOTE'):
            X = data_self_attended.drop('Vulnerability_Truth', axis =1)
            y = data_self_attended['Vulnerability_Truth']

            smote = imblearn.over_sampling.SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            y_resampled_ = np.reshape(y_resampled,(-1, y_resampled.size)).transpose()

            # print(X_resampled.shape,y_resampled_.shape)
            # print(len(X_resampled),len(y_resampled_))

            data_balanced.append(header)

            for i in range(len(X_resampled)):
                vulnerabilityTruth = y_resampled_[i][0]
                
                u_input = []
                if("A" in attcombination):
                    a = X_resampled['A'][i]
                    u_input.append(a)
                if("B" in attcombination):
                    b = X_resampled['B'][i]
                    u_input.append(b)
                if("C" in attcombination):
                    c = X_resampled['C'][i]
                    u_input.append(c)
                if("D" in attcombination):
                    d = X_resampled['D'][i]
                    u_input.append(d)
                # ...extend or delete depending on the attributes you choose to evaluate                        

                u_input.append(vulnerabilityTruth) 

                data_balanced.append(u_input)
            
            positives_ = 0
            negatives_ = 0
            for i in range(len(data_balanced)):
                st = data_balanced[i][len(data_balanced[i])-1]
                if(st == 1):
                    positives_ = positives_+1
                elif(st ==0):
                    negatives_  = negatives_ +1
                    
            info = 'class distribution ratio (vulnerability vs non-vulnerability)-'+_balancing_type+':'+str(positives_)+'/'+str(negatives_)
            
            class_distribution_ratio_ = negatives_/positives_

            postbalance = {
                "BalancingType":"SMOTE",
                "ClassDistributionRatio":class_distribution_ratio_,
                "NoVulnerability":positives_,
                "NoNonVulnerability":negatives_
            }
            configuration['PostBalance'] = postbalance
    
            # save data
            if not os.path.exists(f2_balanced): 
                with open(f2_balanced, 'w') as file: 
                    pass
            df = pd.DataFrame(data_balanced)
            df.to_csv(f2_balanced, index=False, header=False)

        elif(_balancing_type == 'RANDOMOVERSAMPLING'):
            majority_class_label = None
            minority_class_label = None

            if(positives >negatives):
                majority_class_label = 1
                minority_class_label = 0
            else:
                majority_class_label = 0
                minority_class_label = 1
            
            minority_class = data_self_attended[data_self_attended['Vulnerability_Truth'] == minority_class_label]
            majority_class = data_self_attended[data_self_attended['Vulnerability_Truth'] == majority_class_label]
            
            # Upsample the minority class (random oversampling)
            minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
            # Combine the upsampled minority class with the majority class
            data_balanced = pd.concat([majority_class, minority_upsampled])
            # 
            positives_ = len(data_balanced[data_balanced['Vulnerability_Truth'] == 1])
            negatives_ = len(data_balanced[data_balanced['Vulnerability_Truth'] == 0])
            info = 'class distribution ratio (vulnerability vs non-vulnerability)-' +_balancing_type+':'+str(positives_)+'/'+str(negatives_)
            
            class_distribution_ratio_ = negatives_/positives_

            postbalance = {
                "BalancingType":"RANDOMOVERSAMPLING",
                "ClassDistributionRatio":class_distribution_ratio_,
                "NoVulnerability":positives_,
                "NoNonVulnerability":negatives_
            }
            configuration['PostBalance'] = postbalance

             # Save the data
            # if not os.path.exists(f2_balanced): 
            #     with open(f2_balanced, 'w') as file: 
            #         pass
                
            # with open(f2_balanced, "w") as f:
            #     f.write("\n".join(str(data_balanced)))
            if not os.path.exists(f2_balanced): 
                with open(f2_balanced, 'w') as file: 
                    pass
            df = pd.DataFrame(data_balanced)
            df.to_csv(f2_balanced, index=False, header=False)



        elif(_balancing_type == 'RANDOMUNDERSAMPLING'):
            majority_class_label = None
            minority_class_label = None

            if(positives >negatives):
                majority_class_label = 1
                minority_class_label = 0
            else:
                majority_class_label = 0
                minority_class_label = 1
            
            minority_class = data_self_attended[data_self_attended['Vulnerability_Truth'] == minority_class_label]
            majority_class = data_self_attended[data_self_attended['Vulnerability_Truth'] == majority_class_label]
            
            # Downsample the majority class (random undersampling)
            majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
            # Combine the downsampled majority class with the minority class
            data_balanced = pd.concat([minority_class, majority_downsampled])
            # 
            positives_ = len(data_balanced[data_balanced['Vulnerability_Truth'] == 1])
            negatives_ = len(data_balanced[data_balanced['Vulnerability_Truth'] == 0])
            info = 'class distribution ratio (vulnerability vs non-vulnerability)-' +_balancing_type+':'+str(positives_)+'/'+str(negatives_)
            
            class_distribution_ratio_ = negatives_/positives_
            postbalance = {
                "BalancingType":"RANDOMUNDERSAMPLING",
                "ClassDistributionRatio":class_distribution_ratio_,
                "NoVulnerability":positives_,
                "NoNonVulnerability":negatives_
            }
            configuration['PostBalance'] = postbalance
            
             # Save the data
            # if not os.path.exists(f2_balanced): 
            #     with open(f2_balanced, 'w') as file: 
            #         pass
                
            # with open(f2_balanced, "w") as f:
            #     f.write("\n".join(str(data_balanced)))            
            if not os.path.exists(f2_balanced): 
                with open(f2_balanced, 'w') as file: 
                    pass
            df = pd.DataFrame(data_balanced)
            df.to_csv(f2_balanced, index=False, header=False)

        return data_balanced

    def createDirs(self, directory):
        shutil.rmtree(directory, ignore_errors=True)

        if not os.path.exists(directory):
            os.makedirs(directory)

    #read and remove headers from data
    def removeHeader(self,f1,f2):
        with open(f1, "r") as f:
            data = f.read().split("\n")
        
        # Remove the 1st line
        del data[0]

        # Save the data
        if not os.path.exists(f2): 
            with open(f2, 'w') as file: 
                pass
            
        with open(f2, "w") as f:
            f.write("\n".join(data))

    def splitTrainTestData(self,all_file,train_file,test_file):
        df = pd.read_csv(all_file)
        indices = np.arange(len(df))
        indices_train, indices_test = train_test_split(indices, test_size=0.2)
        df_train = df.iloc[indices_train]
        df_test = df.iloc[indices_test]

        df_train.to_csv(train_file)
        df_test.to_csv(test_file)

    
    def configexist(configuration):
        exist = False
        
        label1 = Utilities.getLabel(configuration)

        # check completed configurations
        performancedir = "data/_performance"+str(configuration["epochs"])

        dirExist = os.path.exists(performancedir)
        if (dirExist):
            pfiles = [f for f in listdir(performancedir) if isfile(join(performancedir, f))]

            for pfile in pfiles:
                configuration = Utilities.readconfigPerformance(performancedir,pfile)
                label2 = Utilities.getLabel(configuration)
                
                if(label1 == label2):
                    exist = True
                    break
        if(exist):
            return configuration
        else:
            return None

    def readconfigPerformance(dir, fileName):
        configuration = None
        with open(dir+"/"+fileName, 'r') as openfile: 
            # Reading from json file
            configuration = json.load(openfile)
        return configuration
   
    
    def getLabel(configuration):
        attmetrics= []
        if("A" in configuration['attcombination']):
            attmetrics.append('A') # age            
        if("B" in configuration['attcombination']):
            attmetrics.append('B') # number of issues         
        if("C" in configuration['attcombination']):
            attmetrics.append('C') # difference in date between last repository update and last issue           
        if("D" in configuration['attcombination']):
            attmetrics.append('D') # regulatory authority                 
        # ...extend or delete depending on the attributes you choose to evaluate

        attmetrics_str = ','.join(str(x) for x in attmetrics)

        label ='Data:[epochs:'+str(configuration['epochs'])+', BT:'+str(configuration["BalancingType"])+', ATT:['+attmetrics_str+']'

        return label
    
    def writeConfigPerformance(configuration):
        performancedir = "data/_performance"+str(configuration["epochs"])
        isExist = os.path.exists(performancedir)
        if not isExist:
            os.makedirs(performancedir)
        uid = Utilities.getUniqueId(configuration)
        performance_file = "data/_performance"+str(configuration["epochs"])+"/p_"+uid+".json"
        
        # Serializing json
        json_object = json.dumps(configuration, indent=2)
        
        # Writing to performance dir
        with open(performance_file, "w") as outfile:
            outfile.write(json_object)
        
        pass 


    def getUniqueId(configuration):
        attmetrics= []
        if("A" in configuration['attcombination']):
            attmetrics.append('A') # age            
        if("B" in configuration['attcombination']):
            attmetrics.append('B') # number of issues         
        if("C" in configuration['attcombination']):
            attmetrics.append('C') # difference in date between last repository update and last issue           
        if("D" in configuration['attcombination']):
            attmetrics.append('D') # regulatory authority     
        # ...extend or delete depending on the attributes you choose to evaluate

        attmetrics_str = ''.join(str(x) for x in attmetrics)
        uid =str(configuration['epochs'])+attmetrics_str+str(configuration["BalancingType"])
        
        return uid
    

    def prettyprintconfig(configuration, appendtofile):

        configconvert = {
            "epochs":configuration["epochs"],
            # "BT":configuration["BalancingType"],
            "preBal":str(round(configuration["PreBalance"]["ClassDistributionRatio"],3))+":"+str(round(configuration["PreBalance"]["NoVulnerability"],3))+"/"+str(round(configuration["PreBalance"]["NoNonVulnerability"],3)),
            "postBal":configuration["PostBalance"]['BalancingType']+"("+str(round(configuration["PostBalance"]["ClassDistributionRatio"],3))+":"+str(round(configuration["PostBalance"]["NoVulnerability"],3))+"/"+str(round(configuration["PostBalance"]["NoNonVulnerability"],3))+")",
            "atts":configuration["attcombination"],
            "TPR":round(configuration["performance"]["truepositiverate"],3),
            "FPR":round(configuration["performance"]["falsepositiverate"],3),
            "P":round(configuration["performance"]["precision"],3),
            "R":round(configuration["performance"]["recall"],3),
            "A":round(configuration["performance"]["accuracy"],3),
            "F1":round(configuration["performance"]["f1score"],3)
        } 

        print('\r' + json.dumps(configconvert), end='')

        if(appendtofile):
            resultsfile = "data/_results"+str(configuration["epochs"])+"/results.json"
            f = open(resultsfile, "a")
            f.write(json.dumps(configconvert))
            f.close()

            row = []
            row.append(configuration["InputNodes"])
            row.append(configuration["LearningRate"])
            row.append(configuration["HiddenNodes"])
            row.append(configuration["OutputNodes"])
            row.append(configuration["epochs"])
            row.append(configuration["PreBalance"]["ClassDistributionRatio"])
            row.append(configuration["PreBalance"]["NoVulnerability"])
            row.append(configuration["PreBalance"]["NoNonVulnerability"])
            row.append(configuration["PostBalance"]['BalancingType'])
            row.append(configuration["PostBalance"]["ClassDistributionRatio"])
            row.append(configuration["PostBalance"]["NoVulnerability"])
            row.append(configuration["PostBalance"]["NoNonVulnerability"])
            row.append(configuration["attcombination"])
            row.append(configuration["performance"]["truepositiverate"])
            row.append(configuration["performance"]["falsepositiverate"])
            row.append(configuration["performance"]["precision"])
            row.append(configuration["performance"]["recall"])
            row.append(configuration["performance"]["accuracy"])
            row.append(configuration["performance"]["f1score"])
        
            with open("data/_results"+str(configuration["epochs"])+"/results.csv", 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(row)
        pass


def main():
    # number of input, hidden and output nodes
    input_nodes = 5
    hidden_nodes = 2
    output_nodes = 2 

    # learning rate
    learning_rate = 0.1

    # data reuse
    max_epochs = 10

    pheader = []
    pheader.append("InputNodes")
    pheader.append("LearningRate")
    pheader.append("HiddenNodes")
    pheader.append("OutputNodes")
    pheader.append("epochs")
    pheader.append("PreBalance:ClassDistributionRatio")
    pheader.append("BalancingType")
    pheader.append("PostBalance:ClassDistributionRatio")
    pheader.append("truepositiverate")
    pheader.append("falsepositiverate")
    pheader.append("precision")
    pheader.append("recall")
    pheader.append("accuracy")
    pheader.append("f1score")

    for epochs in range(1,max_epochs):
        resultsdir = "data/_results"+str(epochs)
        headlessdir = "data/_headless"+str(epochs)
        trainingdir = "data/_training"+str(epochs)
        testingdir = "data/_testing"+str(epochs)
        balanceddir = "data/_balanced"+str(epochs)
        attendeddir = "data/_attended"+str(epochs)

        Utilities().createDirs(resultsdir)
        Utilities().createDirs(headlessdir)
        Utilities().createDirs(trainingdir)
        Utilities().createDirs(testingdir)
        Utilities().createDirs(balanceddir)
        Utilities().createDirs(attendeddir)

        with open(resultsdir+'/results.csv', 'w') as f:
           csv_writer = csv.writer(f)
           csv_writer.writerow(pheader)

        # process = Process(target=processInstance, args=(epochs,input_nodes,hidden_nodes,output_nodes,learning_rate,))
        # process.start()
        processInstance(epochs,input_nodes,hidden_nodes,output_nodes,learning_rate)
        pass

if __name__ == "__main__":
    main()
# 