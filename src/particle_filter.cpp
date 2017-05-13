/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <assert.h>
#include <limits>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 1000;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    Particle p;
    for (int i = 0; i < num_particles; ++i) {
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    assert(yaw_rate != 0);
    for (auto it = particles.begin(); it != particles.end(); ++it) {
        Particle &p = *it;
        // Calculate particle motion
        double yaw_delta = yaw_rate * delta_t;
        p.x += velocity / yaw_rate * (sin(p.theta + yaw_delta) - sin(p.theta));
        p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_delta));
        p.theta += yaw_delta;
        while (p.theta < -M_PI) p.theta += 2 * M_PI;
        while (p.theta > M_PI) p.theta -= 2 * M_PI;
        // Add noise to location
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        p.x = dist_x(gen);
        p.y = dist_y(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    for (auto &obs: observations) {
        obs.id = -1;
        double closest_dist = numeric_limits<double>::max();
        // Find the prediction that is closest to this observation
        for (int i = 0; i < predicted.size(); ++i) {
            const LandmarkObs &pred = predicted.at(i);
            double d = dist(obs.x, obs.y, pred.x, pred.y);
            if (d < closest_dist) {
                obs.id = i;
                closest_dist = d;
            }
        }
        assert(obs.id > -1);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html
    weights.clear();
    for (auto &p: particles) {
        // Create a vector of predicted landmark observations
        std::vector<LandmarkObs> predicted;
        for (auto const &l: map_landmarks.landmark_list) {
            // Compare landmark distance to sensor range
            if (dist(p.x, p.y, l.x_f, l.y_f) > sensor_range) {
                // The landmark is too far away, skip it
                continue;
            }
            LandmarkObs l_obs{l.id_i, l.x_f, l.y_f};
            predicted.push_back(l_obs);
        }
        assert(predicted.size());
        // Transform observations to map coordinates
        std::vector<LandmarkObs> map_observations;
        for (auto const &obs: observations) {
            LandmarkObs m_obs;
            // Note: contrary to the comment above, the minus sign here seems to be needed, 
            m_obs.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            m_obs.y = p.y + obs.y * cos(p.theta) + obs.x * sin(p.theta);
            map_observations.push_back(m_obs);
        }
        dataAssociation(predicted, map_observations);
        // Calculate particle's weight
        p.weight = 1.0;
        for (auto const &obs: map_observations) {
            // Get the closest landmark to this observation
            const LandmarkObs &l = predicted.at(obs.id);
            // Calculate observation's weight using bivariate Gaussian probability distribution
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            double x_diff = obs.x - l.x;
            double y_diff = obs.y - l.y;
            double w = 1.0 / (2.0 * M_PI * std_x * std_y) * exp(-(x_diff * x_diff / (2.0 * std_x * std_x) + y_diff * y_diff / (2.0 * std_y * std_y)));
            p.weight *= w;
        }
        // Store the weights separately for easier use during resampling
        weights.push_back(p.weight);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> new_particles;
    // Create a discrete distribution based on the particle weights
    discrete_distribution<int> dist(weights.begin(), weights.end());
    for (int i = 0; i < particles.size(); ++i) {
        int index = dist(gen);
        new_particles.push_back(particles.at(index));
    }
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
