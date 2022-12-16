import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class App {
    public static final int INPUT_NEURON_COUNT = 55;
    // Input languages: Spanish, French, Italian, Romanian, Portuguese
    public static final String[][] DICTIONARY_INPUT_TRAINING = new String[][]{
            {"deporte"}, {"sport"}, {"sport"}, {"sportiv"}, {"esporte"},
            {"raqueta"}, {"raquette"}, {"racchetta"}, {"racheta"}, {"raquete"},
            {"jugador"}, {"joueur"}, {"giocatore"}, {"jucator"}, {"jogador"},
            {"gol"}, {"objectif"}, {"obbiettivo"}, {"poarta"}, {"objetivo"},
            {"atleta"}, {"athlete"}, {"atleta"}, {"atlet"}, {"atleta"},
            {"casco"}, {"casque"}, {"casco"}, {"casca"}, {"capacete"},
            {"guante"}, {"gant"}, {"guanto"}, {"manusa"}, {"luva"},
            {"dardos"}, {"flechettes"}, {"freccette"}, {"sageti"}, {"dardos"},
            {"esqui"}, {"ski"}, {"sciare"}, {"schi"}, {"esqui"},
            {"nadar"}, {"nager"}, {"nuotare"}, {"inot"}, {"nadar"},
            {"hockey"}, {"le hockey"}, {"hockey"}, {"hochei"}, {"hoquei"},
            {"futbol"}, {"football"}, {"calcio"}, {"fotbal"}, {"futebol"},
            {"aro"}, {"cerceau"}, {"cerchio"}, {"cerc"}, {"aro"},
            {"carrera"}, {"course"}, {"gara"}, {"rasa"}, {"corrida"},
            {"torneo"}, {"tournoi"}, {"torneo"}, {"campionat"}, {"torneio"},
            {"copa"}, {"gobelet"}, {"calice"}, {"pocal"}, {"calice"},
            {"medalla"}, {"medaille"}, {"medaglia"}, {"medalie"}, {"medalha"}
    };

    public static final String[][] DICTIONARY_IDEAL = new String[][]{
            {"sport"}, {"sport"}, {"sport"}, {"sport"}, {"sport"},
            {"reket"}, {"reket"}, {"reket"}, {"reket"}, {"reket"},
            {"igrač"}, {"igrač"}, {"igrač"}, {"igrač"}, {"igrač"},
            {"gol"}, {"gol"}, {"gol"}, {"gol"}, {"gol"},
            {"atletičar"}, {"atletičar"}, {"atletičar"}, {"atletičar"}, {"atletičar"},
            {"kaciga"}, {"kaciga"}, {"kaciga"}, {"kaciga"}, {"kaciga"},
            {"rukavica"}, {"rukavica"}, {"rukavica"}, {"rukavica"}, {"rukavica"},
            {"pikado"}, {"pikado"}, {"pikado"}, {"pikado"}, {"pikado"},
            {"skijati"}, {"skijati"}, {"skijati"}, {"skijati"}, {"skijati"},
            {"plivati"}, {"plivati"}, {"plivati"}, {"plivati"}, {"plivati"},
            {"hokej"},{"hokej"},{"hokej"},{"hokej"},{"hokej"},
            {"nogomet"},{"nogomet"},{"nogomet"},{"nogomet"},{"nogomet"},
            {"obruč"},{"obruč"},{"obruč"},{"obruč"},{"obruč"},
            {"utrka"},{"utrka"},{"utrka"},{"utrka"},{"utrka"},
            {"turnir"},{"turnir"},{"turnir"},{"turnir"},{"turnir"},
            {"pehar"},{"pehar"},{"pehar"},{"pehar"},{"pehar"},
            {"medalja"},{"medalja"},{"medalja"},{"medalja"},{"medalja"}
    };

    public static final String[][] DICTIONARY_TEST = new String[][]{
            {"esport"}, //sport in Catalan
            {"rakieta"}, //racket in Polish
            {"obxectivo"}, //goal in Galician
            {"atlet"}, //athlete in Danish
            {"sci"}, //ski in Corsican
            {"nata"}, //swim in Corsican
            {"medalgia"}, //typo in Italian word "medaglia"
            {"gobetet"}, //typo in French word "gobelet"
    };

    public static void main(String[] args) {
        double[][] normalizedInput = Normalizer.normalizeInput(DICTIONARY_INPUT_TRAINING);
        double[][] normalizedIdeal = Normalizer.normalizeIdeal(DICTIONARY_IDEAL);

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(new ActivationElliott(),false, INPUT_NEURON_COUNT));
        network.addLayer(new BasicLayer(new ActivationElliott(),false,50));
        network.addLayer(new BasicLayer(new ActivationElliott(),false,50));
        network.addLayer(new BasicLayer(new ActivationElliott(),false,normalizedIdeal[0].length));
        network.getStructure().finalizeStructure();
        network.reset();

        MLDataSet trainingSet = new BasicMLDataSet(normalizedInput, normalizedIdeal);
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet, 0.7, 0.3);

        int epoch = 0;
        do{
            train.iteration();
            System.out.printf("Pogreška nakon %d. iteracije iznosi %f\n",epoch,train.getError());
            epoch++;
        } while(train.getError() > 0.001);

        System.out.println("Training results: ");
        for (int i = 0; i < trainingSet.size(); i++) {
            MLDataPair pair = trainingSet.get(i);
            final MLData output = network.compute(pair.getInput());
            System.out.println(DICTIONARY_INPUT_TRAINING[i][0]+" ("+DICTIONARY_IDEAL[i][0]+")");
            calculateValue(output);
            System.out.println("--------------------------");
        }

        System.out.format("\n%15s%15s%15s%n","input", "output", "certainty");
        System.out.println("----------------------------------------------------");
        MLDataSet testing = new BasicMLDataSet(Normalizer.normalizeInput(DICTIONARY_TEST), null);
        for (int i = 0; i < testing.size(); i++) {
            MLDataPair pair = testing.get(i);
            final MLData output = network.compute(pair.getInput());
            calculateValue(output, DICTIONARY_TEST[i][0]);
        }

        System.out.println("Ukupan broj epoha: "+epoch);
    }

    public static void calculateValue(MLData output){
        boolean found = false;
        double total = 0;
        for(int i=0; i<output.size(); i++){
            total += output.getData(i);
        }
        for (int i = 0; i < output.size(); i++) {
            double value = output.getData(i);
            double likelyhood = value/total;
            if(output.getData(i) > 0.1){
                found = true;
                double certainty = (output.getData(i)*100/total);
                System.out.printf("I am %.2f%% sure this is %s%n",certainty, DICTIONARY_IDEAL[i*5][0]);
            }
        }
        if(!found) System.out.println("I do not know what this is.");
    }

    public static void calculateValue(MLData output, String word){
        boolean found = false;
        double total = 0;
        for(int i=0; i<output.size(); i++){
            total += output.getData(i);
        }

        for (int i = 0; i < output.size(); i++) {
            double value = output.getData(i);
            double likelyhood = value/total;
            if(output.getData(i) > 0.1){
                double certainty = (output.getData(i)*100/total);
                System.out.format("%15s%15s%15.2f%%%n",found ? "" : word, DICTIONARY_IDEAL[i*5][0], certainty);
                found = true;
            }
        }
        if(!found) System.out.format("%15s%30s%n", word, "I do not know what this is");
    }
}
