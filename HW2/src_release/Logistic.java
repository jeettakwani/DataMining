import Jama.Matrix;
import Utility.Utility;
import weka.core.*;
import weka.core.Instance;
import weka.core.matrix.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng
 * Date: 6/22/12
 * Time: 11:01 PM
 * To change this template use File | Settings | File Templates.
 */
public class Logistic {

    /** the learning rate */
    private double rate;

    /** the weight to learn */
    private double[] weights;
    
    private double bias; 

    /** the number of iterations */
    private int ITERATIONS = 6000;
    
    private double EPSI = Double.longBitsToDouble(971l << 52);

    public Logistic(int n) {
        this.rate = 0.001;
        weights = new double[n];
        bias = 0;
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public void train(Instances data) {
        double lik = 0.0;
        double newlik = 0.0;
    	for (int n=0; n<ITERATIONS; n++) {

            double []predictVec = new double[data.numInstances()];
            for (int i=0; i<data.numInstances(); i++) {
                Instance x = data.instance(i);
                double predicted = classify(x);
                predictVec[i] = predicted;
            }
	        //update weights and bias
            Matrix hessian = getHessianMatrix(data, weights);
            Matrix lambda =
                    new Matrix(hessian.getRowDimension(), hessian.getColumnDimension()).identity(hessian.getRowDimension(), hessian.getColumnDimension()).times(-0.001);;
            hessian = hessian.plus(lambda);

            double[] first_derivative = first_derivative(data, weights);
            double [][] second_derivative = hessian.inverse().getArray();

            converged(first_derivative);

            double[] intermediate_weights = new double[weights.length];
            for(int i =0; i < weights.length; i++)
            {
                intermediate_weights[i] = 0.0;
                for(int j = 0; j < weights.length; j++)
                {
                    intermediate_weights[i] += second_derivative[i][j]*first_derivative[j];
                }
            }

            for(int i =0; i< weights.length; i++)
            {
                weights[i] = weights[i] - intermediate_weights[i];
            }

            this.bias = weights[0];

	        //calculate log likelihood function
            newlik = maximum_likelihood_est(data,weights,predictVec);
            if(Utils.eq(newlik,lik))
                break;
            lik = newlik;
     
            System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + newlik);
        }
    }

    /** functio to calculate convergence */
    public boolean converged(double[] derivative)
    {
        for(int i = 0; i < derivative.length; i++)
        {
            if(!Utils.eq(derivative[i],0))
                return false;
        }
        return true;
    }


    private double classify(Instance x) {
        double logit = bias;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * x.value(i);
        }
        return sigmoid(logit);
    }

    /** this function calulates the values for hessian matrix
     *
     * @param data Instance data
     * @param beta
     * @return Matrix hessian
     */
    public Matrix getHessianMatrix(Instances data, double [] beta)
    {
        double[][] hessian = new double[beta.length][beta.length];

        for(int i=0; i<beta.length;i++)
        {
            for(int j=0;j<beta.length;j++){

                hessian[i][j] = 0.0;

                for(int k=0; k<data.numInstances();k++)
                {
                    Instance x = data.instance(i);
                    double predictY = classify(x);

                    hessian[i][j] -= x.value(i)*x.value(j)*predictY*(1-predictY);
                }
            }
        }

        Matrix hess = new Matrix(hessian);
        return hess;
    }

    /**
     *  This function calculates the value of first derivative
     * @param data
     * @param beta
     * @return
     */
    public double[] first_derivative(Instances data, double[] beta)
    {
        double[] sum = new double[beta.length];
        for (int j=0;j<beta.length;j++) {


            sum[j] = 0.0;
        }

        for(int i =0; i < data.numInstances(); i++)
        {
            Instance x = data.instance(i);
            double predict_y = classify(x);
            for (int j=0;j<x.numAttributes()-1;j++) {


                sum[j] = sum[j] + x.value(j) * (x.classValue() - predict_y);
            }
        }
        return sum;
    }


    /**
     * function to calculate the mle.
     * @param data
     * @param beta
     * @param predict_vector
     * @return
     */
    public double maximum_likelihood_est(Instances data, double[] beta, double[] predict_vector)
    {
        double sum = 0.0;

        for (int i=0; i<data.numInstances(); i++) {

            Instance x = data.instance(i);
            double yi = predict_vector[i];

            double xTB = classify(x);

            sum += (yi*xTB) - (Math.log(1+Math.exp(xTB))/Math.log(2.0));
        }

        return sum;
    }
    /**
     * calculates the accuracy for test data
     * @param test test data
     * @return accuracy
     */
    public double calculate_accuracy(Instances test)
    {
        double number_rows = test.numInstances();
        double c = 0.0;
        double pred ;

        for(int i=0;i<number_rows;i++){

            Instance x = test.instance(i);
            pred = classify(x);

            if (pred> 0.5)
                pred =1;
            else
                pred =0;

            if(pred == x.classValue())
                c++;
        }

        return c/number_rows;
    }

    /**
     * Initialize beta values to 0.
     */
    public void initialize_beta(){
        for(int i =0; i < weights.length; i++)
            weights[i] = 0.0;
    }


    public static void main(String... args) throws  IOException {
        BufferedReader file = Utility.readFile("data_logistic.arff");
        Instances data = new Instances(file);

        Logistic logistic = new Logistic(data.numAttributes() - 1);

        int class_index = data.numAttributes() - 1;
        int folds = 5;
        Random r = new Random(9);
        Instances random_data = new Instances(data);
        random_data.randomize(r);

        double accuracy = 0.0;

        for (int i = 0; i < folds; i++) {

            Instances train = random_data.trainCV(folds, i);
            Instances test = random_data.testCV(folds, i);

            train.setClassIndex(class_index);
            test.setClassIndex(class_index);
            logistic.initialize_beta();
            logistic.bias = 0.0;
            logistic.train(train);
            accuracy += logistic.calculate_accuracy(test);

        }

        accuracy = accuracy / folds;

        System.out.print("accuracy by 5-fold:" + accuracy*100);

    }
}
