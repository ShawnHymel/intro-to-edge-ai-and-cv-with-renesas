// Sensor channels: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
#define NUM_CHANNELS 6

// Mean for each sensor channel
const float STANDARDIZATION_MEANS[NUM_CHANNELS] = {
    -104.811367f, -367.461367f, 1497.516767f, -27.567033f, 26.882467f, -107.321133f
};

// Standard deviation for each sensor channel
const float STANDARDIZATION_STD_DEVS[NUM_CHANNELS] = {
    715.407469f, 991.475474f, 1131.553970f, 1132.640657f, 1511.256275f, 1366.474267f
};
