package LinearRegression;

import Jama.Matrix;
import weka.core.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Simple Linear Regression implementation
 */
public class LinearRegression {
    public static void linearRegression(int choice) throws Exception {
        Matrix trainingData = MatrixData.getDataMatrix("data/linear_regression/linear-regression-train.csv");
        // getMatrix(Initial row index, Final row index, Initial column index, Final column index)
        Matrix train_x = trainingData.getMatrix(0, trainingData.getRowDimension() - 1, 0, trainingData.getColumnDimension() - 2);
        Matrix train_y = trainingData.getMatrix(0, trainingData.getRowDimension() - 1, trainingData.getColumnDimension() - 1, trainingData.getColumnDimension() - 1);

        Matrix testData = MatrixData.getDataMatrix("data/linear_regression/linear-regression-test.csv");
        Matrix test_x = testData.getMatrix(0, testData.getRowDimension() - 1, 0, testData.getColumnDimension() - 2);
        Matrix test_y = testData.getMatrix(0, testData.getRowDimension() - 1, testData.getColumnDimension() - 1, testData.getColumnDimension() - 1);

        if(choice == 1)
        {
            linear_regression(train_x, train_y, test_x, test_y);
        }
        else
        {
            linear_regression_normalizing_data(train_x, train_y, test_x, test_y);

        }

    }

    public static void linear_regression(Matrix train_x, Matrix train_y, Matrix test_x, Matrix test_y) throws Exception
    {
        Matrix beta = getBeta(train_x, train_y);
        Matrix predictedY = test_x.times(beta);
        double mse = calculateMeanSquareError(predictedY, test_y);
        printOutput(predictedY,beta,mse);

        Matrix beta_for_online_update = getOnlineUpdateBeta(train_x, train_y);
        Matrix predictedY_for_online_update = test_x.times(beta_for_online_update);
        double mse_for_online = calculateMeanSquareError(predictedY_for_online_update, test_y);
        printOutput_for_online(predictedY_for_online_update,beta_for_online_update,mse_for_online);
    }
    public static void linear_regression_normalizing_data(Matrix train_x, Matrix train_y, Matrix test_x, Matrix test_y) throws Exception
    {
        Matrix mean_train = findMean(train_x);
        Matrix standard_deviation = findDeviation(mean_train, train_x);
        Matrix normalize_x = nomarlization(train_x, mean_train, standard_deviation);

        /* Linear Regression */
        /* 2 step process */
        // 1) find beta
        Matrix beta = getBeta(normalize_x, train_y);
        Matrix beta_online_update = getOnlineUpdateBeta(normalize_x, train_y);

        Matrix mean = findMean(test_x);
        Matrix std = findDeviation(mean, test_x);
        Matrix normalize_X = nomarlization(test_x, mean, std);

        // 2) predict y for test data using beta calculated from train data
        Matrix predictedY = normalize_X.times(beta);
        Matrix predictedY_online_update = normalize_X.times(beta_online_update);

        double mse = calculateMeanSquareError(predictedY, test_y);
        double mse_for_online_update = calculateMeanSquareError(predictedY_online_update, test_y);
        printOutput(predictedY, beta, mse);
        printOutput_for_online(predictedY_online_update,beta_online_update,mse_for_online_update);
    }


    /**  @params: X and Y matrix of training data
     * returns value of beta calculated using the formula beta = (X^T*X)^ -1)*(X^T*Y)
     */
    private static Matrix getBeta(Matrix trainX, Matrix trainY) {
    
    	/****************Please Fill Missing Lines Here*****************/
        Matrix beta = (((trainX.transpose()).times(trainX)).inverse()).times((trainX.transpose()).times(trainY));
        for (int row =0; row<beta.getRowDimension(); row++) {
            //System.out.println(String.valueOf(beta.get(row, 0)));
        }
        System.out.println("------- END OF 1ST METHOD---------");
        return beta;
    }

    /**
     * to calculate beta for online update method
     * @params: X and Y matrix of training data
     * returns value of beta calculated using the formula beta =
     */
    private static Matrix getOnlineUpdateBeta(Matrix x, Matrix y)
    {
        Matrix beta_old = null;
        Matrix b =  Matrix.random(x.getColumnDimension(), 1);
        do {

            for (int i = 0; i < x.getRowDimension(); i++) {
                beta_old = b.getMatrix(0,b.getRowDimension()-1,0,0);
                Matrix current_x = x.getMatrix(i, i, 0, x.getColumnDimension() - 1);
                Matrix current_y = y.getMatrix(i, i, 0, 0);
                Matrix temp = current_x.times(beta_old);
                Matrix something = (current_y.minus(temp).times(2 * 0.001));
                Matrix transpose = (something.times(current_x)).transpose();
                b = beta_old.plus(transpose).getMatrix(0, beta_old.getRowDimension() - 1, 0, beta_old.getColumnDimension() - 1);
                if (convergence(beta_old,b)) {
                    break;
                }

                //beta_old = newbeta.getMatrix(0, newbeta.getRowDimension() - 1, 0, newbeta.getColumnDimension() - 1);
            }
        }while(!convergence(beta_old, b));;
        return b;
    }

    public static boolean convergence(Matrix beta_old, Matrix beta)
    {
        for (int r = 0; r < beta_old.getRowDimension(); r++)
        {
            if(!Utils.eq(beta_old.get(r,0),beta.get(r,0)))
                return false;
        }
        return true;
    }
    /**
     * @params: predicted Y matrix
     * outputs the predicted y values to the text file named "linear-regression-output"
     */
    public static void printOutput(Matrix predictedY, Matrix beta, double mse) throws IOException {
        FileWriter fStream = new FileWriter("output/linear_regression/linear-regression-output.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        out.write("Predicted Y output :");
        out.newLine();
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        out.newLine();
        out.write("Beta value:");
        out.newLine();
        for (int row =0; row<beta.getRowDimension(); row++) {
            out.write(String.valueOf(beta.get(row, 0)));
            out.newLine();
        }
        out.newLine();
        out.write("Mean square error for online update method:");
        out.newLine();
        out.write(String.valueOf(mse));
        out.close();
    }

    /**
     * @params: predicted Y matrix
     * outputs the predicted y values to the text file named "linear-regression-output"
     */
    public static void printOutput_for_online(Matrix predictedY, Matrix beta, double mse) throws IOException {
        FileWriter fStream = new FileWriter("output/linear_regression/linear-regression-output-online.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        out.write("Predicted Y output :");
        out.newLine();
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        out.newLine();
        out.write("Beta value:");
        out.newLine();
        for (int row =0; row<beta.getRowDimension(); row++) {
            out.write(String.valueOf(beta.get(row, 0)));
            out.newLine();
        }
        out.newLine();
        out.write("Mean square error for online update method:");
        out.newLine();
        out.write(String.valueOf(mse));
        out.close();
    }

    /**
     * calculating mean square error for closed
     */
    public static double calculateMeanSquareError(Matrix predictedY, Matrix test_y)
    {
        double sum = 0;
        double mean = predictedY.getRowDimension();
        //System.out.println(mean);
        for(int row = 0; row < predictedY.getRowDimension(); row++)
        {
            double square = (predictedY.get(row,0) - test_y.get(row,0))*(predictedY.get(row,0) - test_y.get(row,0));
            sum = sum + square;
        }
        double mse = sum/mean;
        System.out.println(mse);
        return mse;
    }

    /**
     * find Mean of the given data
     */
    public static Matrix findMean(Matrix x)
    {
        Matrix mean = new Matrix(1, x.getColumnDimension());
        for(int column = 1; column < x.getColumnDimension(); column++) {
            double sum = 0;
            for (int row = 0; row < x.getRowDimension(); row++) {
                sum = sum + x.get(row, column);
            }
            mean.set(0, column, (sum / x.getRowDimension()));
        }
        return mean;
    }

    /**
     * find standard deviation
     */
    public static Matrix findDeviation(Matrix mean, Matrix x) {
        double sum = 0;
        Matrix std = new Matrix(1, x.getColumnDimension());
        for (int column = 1; column < x.getColumnDimension(); column++) {
            sum = 0;
            for (int row = 0; row < x.getRowDimension(); row++) {
                sum = sum + ((x.get(row, column) - mean.get(0,column)) * (x.get(row, column) - mean.get(0,column)));
            }
            std.set(0,column, Math.sqrt(sum/(x.getRowDimension()-1)));
        }
        return std;
    }

    /**
     * Normalization
     */
    public static Matrix nomarlization(Matrix x, Matrix mean, Matrix std) {

        Matrix new_x = x.getMatrix(0, x.getRowDimension() - 1, 0, x.getColumnDimension() - 1);
        for (int column = 1; column < x.getColumnDimension(); column++) {
            for (int row = 0; row < x.getRowDimension(); row++) {
                new_x.set(row, column, ((x.get(row, column) - mean.get(0, column)) / std.get(0, column)));
            }
        }
        return new_x;
    }
}
