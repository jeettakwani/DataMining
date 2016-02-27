package DecisionTree;

import java.io.*;
import java.util.*;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import Utility.Utility;

/*Class for constructing an unpruned decision tree based on
the ID3 algorithm. Can only deal with nominal attributes.
No missing values allowed. Empty leaves may result in unclassified instances.
 */
public class DecisionTree  {

    //The node's successors.
    private DecisionTree[] m_Successors;
    //Attribute used for splitting.
    private Attribute m_Attribute;
    //Class value if node is leaf.
    private double m_ClassValue;
    //Class distribution if node is leaf.
    private double[] m_Distribution;
    // Class attribute of data set.
    private Attribute m_ClassAttribute;

    public DecisionTree() {
    }
    //Builds decision tree classifier.
    public void buildClassifier(Instances data,int choice) throws Exception {
        data = new Instances(data);
        this.makeTree(data, choice);
    }

    private void makeTree(Instances data, int choice) throws Exception {
        if(data.numInstances() == 0) {
            this.m_Attribute = null;
            this.m_ClassValue = Instance.missingValue();
            this.m_Distribution = new double[data.numClasses()];
        } else {
            double[] infoGains = new double[data.numAttributes()];
            double[] gainRation = new double[data.numAttributes()];

            Attribute splitData;
            for(Enumeration attEnum = data.enumerateAttributes(); attEnum.hasMoreElements(); infoGains[splitData.index()] = this.computeInfoGain(data, splitData)) {
                splitData = (Attribute)attEnum.nextElement();
                gainRation[splitData.index()] = this.compute_gain_info(data, splitData);
            }
            if(choice == 1)
            {
                this.m_Attribute = data.attribute(Utils.maxIndex(infoGains));
            }
            else {
                this.m_Attribute = data.attribute(Utils.maxIndex(gainRation));
            }
            if(Utils.eq(infoGains[this.m_Attribute.index()], 0.0D)) {
                this.m_Attribute = null;
                this.m_Distribution = new double[data.numClasses()];

                Instance j;
                for(Enumeration var6 = data.enumerateInstances(); var6.hasMoreElements(); ++this.m_Distribution[(int)j.classValue()]) {
                    j = (Instance)var6.nextElement();
                }

                Utils.normalize(this.m_Distribution);
                this.m_ClassValue = (double)Utils.maxIndex(this.m_Distribution);
                this.m_ClassAttribute = data.classAttribute();
            } else {
                Instances[] var7 = this.splitData(data, this.m_Attribute);
                this.m_Successors = new DecisionTree[this.m_Attribute.numValues()];

                for(int var8 = 0; var8 < this.m_Attribute.numValues(); ++var8) {
                    this.m_Successors[var8] = new DecisionTree();
                    this.m_Successors[var8].makeTree(var7[var8],choice);
                }
            }
        }
    }
    //Classifies a given test instance using the decision tree.
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if(instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("DecisionTree: no missing values, please.");
        } else {
            return this.m_Attribute == null?this.m_ClassValue:this.m_Successors[(int)instance.value(this.m_Attribute)].classifyInstance(instance);
        }
    }

    public String toString() {
        return this.m_Distribution == null && this.m_Successors == null?"DecisionTree: No model built yet.":"DecisionTree\n\n" + this.toString(0);
    }
    //Computes information gain for an attribute.
    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = this.computeEntropy(data);
        Instances[] splitData = this.splitData(data, att);

        /****************Please Fill Missing Lines Here*****************/

        double info_atribute = 0.0;

        for(int i =0; i<splitData.length; i++)
        {
            info_atribute += ((this.computeEntropy(splitData[i]))*(splitData[i].numInstances()))/(data.numInstances());
        }

        infoGain = infoGain - info_atribute;
        
        return infoGain;
    }
    //Computes the entropy of a dataset.
    private double computeEntropy(Instances data) throws Exception {
        double[] classCounts = new double[data.numClasses()];

        Instance entropy;
        for(Enumeration instEnum = data.enumerateInstances(); instEnum.hasMoreElements(); ++classCounts[(int)entropy.classValue()]) {
            entropy = (Instance)instEnum.nextElement();
        }

        double totalEntropy = 0.0D;
        int classNum = data.numClasses();
        double [] classProbVec = new double[classNum];
        
        for(int j = 0; j < classNum; ++j) {
            if(classCounts[j] > 0.0D) {
                classProbVec[j]= classCounts[j]/data.numInstances();
            }
            else
            	classProbVec[j]=0;
        }
        /****************Please Fill Missing Lines Here*****************/

        for (int i =0; i< classNum; i++)
        {
            if (classProbVec[i] == 0)
            {
                continue;
            }
            else

            totalEntropy += ((-1) * classProbVec[i])*(Math.log(classProbVec[i])/Math.log(2));
        }


        return totalEntropy;

    }

    private double compute_gain_info(Instances data, Attribute attribute) throws Exception
    {
        double splitinfo = 0.0;
        double infoGain = this.computeInfoGain(data, attribute);
        Instances [] splitData = this.splitData(data,attribute);

        double number_of_rows = data.numInstances();

        for(int i =0; i < splitData.length; i++)
        {
            double rows_in_bucket = splitData[i].numInstances();

            if(rows_in_bucket == 0)
            {
                continue;
            }
            splitinfo += (-1)*(Math.log(rows_in_bucket / number_of_rows)/Math.log(2)*(rows_in_bucket/number_of_rows));

        }
        return splitinfo == 0? infoGain : infoGain/splitinfo;
    }
    //Splits a dataset according to the values of a nominal attribute.
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];

        for(int instEnum = 0; instEnum < att.numValues(); ++instEnum) {
            splitData[instEnum] = new Instances(data, data.numInstances());
        }

        Enumeration var6 = data.enumerateInstances();

        while(var6.hasMoreElements()) {
            Instance i = (Instance)var6.nextElement();
            splitData[(int)i.value(att)].add(i);
        }

        for(int var7 = 0; var7 < splitData.length; ++var7) {
            splitData[var7].compactify();
        }
        return splitData;
    }

    private String toString(int level) {
        StringBuffer text = new StringBuffer();
        if(this.m_Attribute == null) {
            if(Instance.isMissingValue(this.m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + this.m_ClassAttribute.value((int)this.m_ClassValue));
            }
        } else {
            for(int j = 0; j < this.m_Attribute.numValues(); ++j) {
                text.append("\n");

                for(int i = 0; i < level; ++i) {
                    text.append("|  ");
                }

                text.append(this.m_Attribute.name() + " = " + this.m_Attribute.value(j));
                text.append(this.m_Successors[j].toString(level + 1));
            }
        }
        return text.toString();
    }


    public void decisionTree(int choice) throws Exception {
        BufferedReader file = Utility.readFile("data/decision_tree/votes.arff");
        Instances data = new Instances(file);
        // keeping this as 0 because the class attribute value is in the first column instead of last,
        // this can be used if the class data is in the last column.data.numAttributes()-1;
        int cIdx=0;
        int folds = 5;
        Random r = new Random(25);
        Instances random_data = new Instances(data);
        random_data.randomize(r);

        double accuracy = 0.00;

        for (int i = 0; i< folds; i++)
        {
            Instances train = random_data.trainCV(folds, i);
            Instances test = random_data.testCV(folds, i);

            train.setClassIndex(cIdx);
            test.setClassIndex(cIdx);

            buildClassifier(train, choice);

            accuracy += accuracy_calculator(test);
            printOutput(test , i, accuracy);
        }
        accuracy = accuracy / folds;
        System.out.println(accuracy);
        data.setClassIndex(cIdx);
        //buildClassifier(data);
        //printOutput(data,accuracy);
    }

    private double accuracy_calculator(Instances test) throws IOException, NoSupportForMissingValuesException
    {
        double count = 0.0;
        double number_of_rows = test.numInstances();
        double prediction;
        double actual_value;

        for(int i =0; i < number_of_rows; i++)
        {
            Instance test_instance = test.instance(i);
            prediction = this.classifyInstance(test_instance);
            actual_value = test_instance.classValue();


            if(prediction == actual_value){
                count++;
            }
        }
        return count/number_of_rows;
    }


    private void printOutput(Instances data,int i, double accuracy) throws IOException, NoSupportForMissingValuesException {
        FileWriter fStream = new FileWriter("output/decision_tree/decision-tree-output.txt", true);     // Output File
        BufferedWriter out = new BufferedWriter(fStream);

        out.newLine();
        out.write("Fold: " + i);
        out.newLine();
        for(int index =0; index<data.numInstances();index++) {
            Instance testRowInstance = data.instance(index);
            double prediction =classifyInstance(testRowInstance);
            out.write(data.classAttribute().value((int)(prediction)));
            out.newLine();
        }
        if(i == 4)
        {
            out.newLine();
            out.write("accuracy:");
            out.newLine();
            out.write(String.valueOf(100*(accuracy/5)));
        }
        out.close();
    }
}
