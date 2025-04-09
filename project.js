let video;
let poseNet;
let poses = [];
let population;
let populationSize = 20;

// Target position for demonstration (center of canvas)
let targetX, targetY;

function setup() {
  createCanvas(900, 550);
  targetX = width / 2;
  targetY = height / 2;
  
  // Setup video capture
  let constraints = {
    video: { facingMode: 'user' },
    audio: false
  };
  video = createCapture(constraints);
  video.size(900, 550);
  video.hide();
  
  // Initialize PoseNet using ml5
  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on('pose', function(results) {
    poses = results;
  });
  
  // Initialize the genetic algorithm population
  population = new Population(populationSize);
}

function modelReady() {
  console.log("PoseNet model loaded!");
}

function draw() {
  background(220);
  image(video, 0, 0, width, height);
  
  // Draw PoseNet keypoints and skeleton
  drawKeypoints();
  drawSkeleton();
  
  // If at least one pose is detected, run the GA using the right wrist position
  if (poses.length > 0) {
    let pose = poses[0].pose;

    // Classify posture
let posture = classifyPosture(pose);

// Display on screen
fill(0);
textSize(24);
text("Posture: " + posture, 10, 30);
 
    let rightWrist = pose.rightWrist;
    
    if (rightWrist) {
      // Calculate the raw error between the right wrist and the target position
      let error = dist(rightWrist.x, rightWrist.y, targetX, targetY);
      
      // Evolve the GA based on the error (each individual applies a correction)
      population.evaluate(error);
      population.selection();
      
      // Get the best individual from this generation
      let best = population.getBest();
      
      // Display the error and correction on the canvas for demonstration
      fill(255, 0, 0);
      noStroke();
      textSize(24);
      text("Raw Error: " + nf(error, 1, 2), 10, height - 50);
      text("Best Correction: " + nf(best.correction, 1, 2), 10, height - 20);
      
      // Optionally, visualize the target point
      fill(0, 0, 255);
      ellipse(targetX, targetY, 15, 15);
    }
  }
}

//posture estimation single pose

 function classifyPosture(pose) {
  const leftWrist = pose.leftWrist;
  const rightWrist = pose.rightWrist;
  const leftShoulder = pose.leftShoulder;
  const rightShoulder = pose.rightShoulder;
  const leftHip = pose.leftHip;
  const rightHip = pose.rightHip;
  const leftKnee = pose.leftKnee;
  const rightKnee = pose.rightKnee;

  if (
    leftWrist.y < leftShoulder.y &&
    rightWrist.y < rightShoulder.y
  ) {
    return "Hands Up";
  }

  if (
    rightWrist.y < rightShoulder.y &&
    leftWrist.y > leftShoulder.y
  ) {
    return "Right Hand Raised";
  }

  if (
    leftWrist.y < leftShoulder.y &&
    rightWrist.y > rightShoulder.y
  ) {
    return "Left Hand Raised";
  }

  if (
    dist(leftWrist.x, leftWrist.y, rightWrist.x, rightWrist.y) < 50 &&
    leftWrist.y > leftShoulder.y &&
    rightWrist.y > rightShoulder.y
  ) {
    return "Arms Crossed";
  }

  if (
    leftKnee.y > leftHip.y &&
    rightKnee.y > rightHip.y &&
    leftShoulder.y < leftHip.y &&
    rightShoulder.y < rightHip.y &&
    abs(leftHip.y - leftKnee.y) < 200 &&
    abs(rightHip.y - rightKnee.y) < 200
  ) {
    return "Sitting";
  }
  

  if (
    abs(leftShoulder.x - leftHip.x) < 20 &&
    abs(rightShoulder.x - rightHip.x) < 20 &&
    leftShoulder.y < leftHip.y &&
    rightShoulder.y < rightHip.y
  ) {
    return "Standing Straight";
  }

  

  // Add more if needed (like T-pose, sitting, etc.)
  return "Unknown";
}
  



// ----- Genetic Algorithm Classes -----

class Individual {
  constructor() {
    // In this demo, each individual holds a single "correction" parameter.
    // This might represent an adjustment to the measured error.
    this.correction = random(-50, 50);
    this.fitness = 0;
  }
  
  // Calculate fitness based on the error after applying the correction.
  // Lower corrected error results in higher fitness.
  calcFitness(error) {
    let correctedError = abs(error + this.correction);
    this.fitness = 1 / (correctedError + 1); // +1 prevents division by zero
  }
  
  // Combine traits from two parents to produce a child.
  crossover(partner) {
    let child = new Individual();
    child.correction = random() < 0.5 ? this.correction : partner.correction;
    return child;
  }
  
  // Apply random mutation to the individual.
  mutate(mutationRate) {
    if (random(1) < mutationRate) {
      this.correction += random(-5, 5);
    }
  }
}

class Population {
  constructor(size) {
    this.individuals = [];
    this.size = size;
    this.mutationRate = 0.1;
    for (let i = 0; i < size; i++) {
      this.individuals.push(new Individual());
    }
  }
  
  // Evaluate all individuals based on the given error.
  evaluate(error) {
    for (let ind of this.individuals) {
      ind.calcFitness(error);
    }
  }
  
  // Select the next generation via a simple mating pool method.
  selection() {
    let matingPool = [];
    // Add individuals to the pool a number of times proportional to their fitness.
    for (let ind of this.individuals) {
      let n = ind.fitness * 100; // Higher fitness gets more copies
      for (let j = 0; j < n; j++) {
        matingPool.push(ind);
      }
    }
    
    let newIndividuals = [];
    for (let i = 0; i < this.size; i++) {
      let parentA = random(matingPool);
      let parentB = random(matingPool);
      let child = parentA.crossover(parentB);
      child.mutate(this.mutationRate);
      newIndividuals.push(child);
    }
    this.individuals = newIndividuals;
  }
  
  // Return the best individual (highest fitness) from the population.
  getBest() {
    let best = this.individuals[0];
    for (let ind of this.individuals) {
      if (ind.fitness > best.fitness) {
        best = ind;
      }
    }
    return best;
  }
}

// ----- PoseNet Helper Functions -----

// Draw ellipses over the detected keypoints.
function drawKeypoints() {
  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      let keypoint = pose.keypoints[j];
      if (keypoint.score > 0.2) {
        fill(0, 255, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

// Draw lines between connected keypoints (skeleton).
function drawSkeleton() {
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(0, 0, 255);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}
