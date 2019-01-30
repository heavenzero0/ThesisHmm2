<template>
  <v-app>
    <v-container>
      <Spinner v-if="loading"/>
      <v-card v-if="!loading">
        <Graph :data="data" :options="chartOptions" type="LineChart"/>
      </v-card>
    </v-container>
  </v-app>
</template>

<script>
import * as tf from "@tensorflow/tfjs";
import dna from "./assets/training.json";
import dnaTesting from "./assets/testing.json";
// import input from "./assets/input.json";
// import output from "./assets/output.json";
import Spinner from "./components/UI/spinner";
import Graph from "./components/Graph/googleGraph";

export default {
  components: {
    Spinner,
    Graph
  },
  data() {
    return {
      trainingData: [],
      outputData: [],
      testingData: [],
      items: [],
      trainingCost: [],
      loading: true,
      data: [["Iterations", "Accuracy"]],
      chartOptions: {
        title: "Hidden Markov Model Accuracy",
        vAxis: { textPosition: "none", minValue: 0 },
        hAxis: { textPosition: "none" },
        legend: { position: "none", maxlines: 3 },
        colors: ["#5870cb"]
      },
      value: []
    };
  },
  methods: {
    train() {
      const hmm = "tanh";
      const model = (this.model = tf.sequential());

      model.add(
        tf.layers.dense({
          inputShape: [2],
          activation: hmm,
          units: 3
        })
      );

      model.add(
        tf.layers.dense({
          inputShape: [3],
          activation: hmm,
          units: 3
        })
      );

      model.add(
        tf.layers.dense({
          activation: hmm,
          units: 3
        })
      );

      model.compile({
        loss: "meanSquaredError",
        optimizer: tf.train.adam(0.06)
      });

      model
        .fit(this.trainingData, this.outputData, { epochs: 50 })
        .then(history => {
          // model.predict(this.testingData).print();
          // const h =
          //   (1 - history.history.loss[history.history.loss.length - 1]) * 100;
          // this.accuracy = Math.round(h * 100) / 100;
          // this.loading = false;
          history.history.loss.forEach((val, index) => {
            const accuracy = (1 - val) * 100;
            const train = [index + 1, Math.round(accuracy * 100) / 100];
            this.value = [...this.value, Math.round(accuracy * 100) / 100];
            this.data = [...this.data, train];
          });
          this.loading = false;
        });
    }
  },
  created() {
    this.trainingData = tf.tensor2d(dna.map(item => [item.x1, item.x2]));
    this.outputData = tf.tensor2d(
      dna.map(item => [
        item.y === "escherichia coli" ? 1 : 0,
        item.y === "pseudomonas_aeriginosa pao1" ? 1 : 0,
        item.y === "Caulobacter vibrioides CB15" ? 1 : 0
      ])
    );
    this.testingData = tf.tensor2d(dnaTesting.map(item => [item.x1, item.x2]));
  },
  mounted() {
    this.train();
    // this.predict();
  }
};
</script>
