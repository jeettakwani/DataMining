import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.util.*;
import java.io.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import Utility.Utility;

public class Knn
{
    /** k = number of nearest neighbors */
    public int k = 0;

    public Knn( int neighbors)
    {
        this.k = neighbors;
    }

    public static void main(String... args) throws FileNotFoundException, IOException {
        BufferedReader file = Utility.readFile("data.arff");
        Instances data = new Instances(file);


        Scanner sc = new Scanner(System.in);
        int neighbors;
        System.out.println("Enter the number of nearest neigbors");
        neighbors = sc.nextInt();
        Knn knn = new Knn(neighbors);
        int cIdx = data.numAttributes()-1;
        int folds = 5;
        Random r = new Random(7);
        Instances random_data = new Instances(data);
        random_data.randomize(r);

        double accuracy = 0.0;

        for(int f=0;f<folds;f++){

            Instances train = random_data.trainCV(folds,f);
            Instances test = random_data.testCV(folds,f);

            train.setClassIndex(cIdx);
            test.setClassIndex(cIdx);



            accuracy += knn.calculate_acc(test, knn.train_x(train, test));

        }

        accuracy = accuracy/folds;

        System.out.print("Accuracy : " + accuracy*100);

    }

    public double calculate_acc(Instances instances, int[] predicted){

        double count = 0.0;

        double size = instances.numInstances();

        for(int i=0;i<instances.numInstances();i++)
        {
            if(instances.instance(i).classValue() == (double)predicted[i])
            {
                count++ ;
            }
        }

        return count/size;
    }

    public  int[] train_x(Instances instances, Instances test) {
        int i =0;
        int[] predicted = new int[test.numInstances()] ;

        for (i = 0; i < test.numInstances(); i++)
        {
            Instance x = test.instance(i);
            predicted[i] = classify(x, instances);
            //System.out.println("Predicted value : " + i + " "+ predicted);
        }
        //int predicted = classify(testpoint, instances);
        //System.out.println("Predicted value : " + predicted);
        return predicted;
    }

    public  int classify(Instance x, Instances instances)
    {
        double[] distance = new  double[instances.numInstances()];
        HashMap<Double, Integer> mapping = new HashMap();
        for(int i = 0; i < instances.numInstances(); i++) {
            double sum = 0.0;
            Instance y = instances.instance(i);
            for (int j = 0; j < instances.numAttributes()-1; j++) {

                double xj = x.value(j);
                double yj = y.value(j);


                sum = sum + (( xj - yj) * (xj - yj));
            }
            distance[i] = Math.sqrt(sum);
            //System.out.println("Distance :" + distance[i]);
            //System.out.println("Label of: " + i +" "  + instances.get(i).getLabel());
            mapping.put(distance[i],(int)y.classValue());
        }
         return find_smallest(mapping);
    }

    public int find_smallest(HashMap<Double,Integer> mapping)
    {
        List keys = new ArrayList<>(mapping.keySet());
        Collections.sort(keys);
        int count_1 =0;
        int count_0 =0;
        //for(int i = 0; i < keys.size(); i++)
            //System.out.println(keys.get(i));
        for (int i = 0; i < this.k; i++)
        {
            //System.out.println("Value of map:" + mapping.get(keys.get(i)));
            if(mapping.get(keys.get(i)) == 1)
                count_1++;
            else
                count_0++;
        }
        if (count_0 < count_1)
            return 1;
        else
            return 0;
    }
}
