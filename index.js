require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');


const { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
	shuffle: true,
	splitTest: 50,
	dataColumns: ['horsepower', 'weight', 'displacement'],
	labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
	learningRate: 0.1,
	iteration: 1,
	batchSize: 10
});
//regression.features.print()
regression.train();

console.log(regression.test(testFeatures, testLabels));

regression.predict([
	[130,1.75,307],
	[96,1.15,122]
]).print();
