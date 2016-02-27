import LinearRegression.LinearRegression;
import DecisionTree.DecisionTree;

import java.util.Scanner;

public class Algorithms {
    public static void main(String args[]) throws Exception {
        System.out.println("\tAlgorithms");
        System.out.println("1) Linear Regression");
        System.out.println("2) Decision Tree");
        System.out.println("3) Exit\n");
        System.out.println("Enter the number corresponding to the algorithm you want to run:");
        Scanner in = new Scanner(System.in);
        int choice = in.nextInt();
        switch(choice){
            case 1: System.out.println("select");
                    System.out.println("1) Non Normalized form");
                    System.out.println("2) Normalized form");
                    Scanner i = new Scanner(System.in);
                    int c = i.nextInt();
                    if (c != 2 && c != 1)
                    {
                        System.exit(1);
                    }
                    LinearRegression lr = new LinearRegression();
                    lr.linearRegression(c);
                    break;
            case 2: System.out.println("select");
                    System.out.println("1) Info Gain");
                    System.out.println("2) Gain Ratio");
                    Scanner j = new Scanner(System.in);
                    int c1 = j.nextInt();
                    if (c1 != 2 && c1 != 1)
                    {
                        System.exit(1);
                    }
                    DecisionTree dt = new DecisionTree();
                    dt.decisionTree(c1);
                    break;
            case 3: System.exit(0);
        }
    }
}
