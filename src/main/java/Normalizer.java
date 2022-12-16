import java.util.ArrayList;
import java.util.List;

public class Normalizer {
    public static double[][] normalizeInput(String[][] input){
        double[][] encodedInput = new double[input.length][App.INPUT_NEURON_COUNT];

        for (int i = 0; i < input.length; i++) {
            String curr = input[i][0];
            encodedInput[i][curr.charAt(0)-'a'] = 1;
            encodedInput[i][curr.charAt(curr.length()-1)+26-'a'] = 1;
            encodedInput[i][52] = (curr.length() / 10)*2-1; //Length
            encodedInput[i][53] = (countVowels(curr)/curr.length())*2-1; //Percentage of vowels
            //Average ASCII value of characters in word
            encodedInput[i][54] = 0;
            for (int j = 1; j < curr.length()-1; j++) {
                encodedInput[i][54] += curr.charAt(j)-'a';
            }
            encodedInput[i][54] /= 8*('z'-'a');
            encodedInput[i][54] = encodedInput[i][54]*2-1;
        }

        //Debug print
        /*for (int i = 0; i < App.INPUT_NEURON_COUNT; i++) {
            if(i<26) System.out.print((char)(i+'a'));
            else System.out.print((char)(i-26+'a'));
            System.out.print("   ");
        }
        System.out.println();
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < App.INPUT_NEURON_COUNT; j++) {
                System.out.print(encodedInput[i][j]+" ");
            }
            System.out.print(encodedInput[i][52]+" ");
            System.out.println(encodedInput[i][53]);
        }*/
        return encodedInput;
    }

    private static double countVowels(String curr) {
        double count = 0;
        for(char c : curr.toCharArray()){
            if("aeiou".indexOf(c) != -1) count++;
        }
        return count;
    }

    public static double[][] normalizeIdeal(String[][] ideal){
        int len = ideal.length;
        List<String> uniques = new ArrayList<>();

        for (int i = 0; i < len; i++) {
            String curr = ideal[i][0];
            if(!uniques.contains(curr)) uniques.add(curr);
        }

        double[][] encodedIdeal = new double[len][uniques.size()];

        for (int i = 0; i < len; i++) {
            String curr = ideal[i][0];
            encodedIdeal[i][uniques.indexOf(curr)] = 1;
        }

        /*for (int i = 0; i < len; i++) {
            for (int j = 0; j < len/5; j++) {
                System.out.print(encodedIdeal[i][j]+" ");
            }
            System.out.println();
        }*/

        return encodedIdeal;
    }
}
