/* Standard libraries */
#include <inttypes.h>

/* Generated code */
#include "hal_data.h"

/* Custom libraries */
#include "common_utils.h"
#include "icm20948/icm20948.h"
#include "inference/model.h"
#include "utils/utils.h"

/******************************************************************************
 * Settings
 ******************************************************************************/

/* Data collection */
#define SAMPLE_RATE_HZ      100
#define SAMPLE_PERIOD_MS    10
#define NUM_SAMPLES         100 /* 1 sec */

/* Sensor channels: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z */
#define NUM_CHANNELS 6

/* Mean for each sensor channel */
const float STANDARDIZATION_MEANS[NUM_CHANNELS] = {
    -104.811367f, -367.461367f, 1497.516767f, -27.567033f, 26.882467f, -107.321133f
};

/* Standard deviation for each sensor channel */
const float STANDARDIZATION_STD_DEVS[NUM_CHANNELS] = {
    715.407469f, 991.475474f, 1131.553970f, 1132.640657f, 1511.256275f, 1366.474267f
};

/* Labels */
#define NUM_CLASSES 5
const char* class_names[] = {
    "_idle",
    "_other",
    "alpha",
    "beta",
    "gamma",
};

/******************************************************************************
 * Data structures
 ******************************************************************************/

/* Sample data structure */
typedef struct {
    uint32_t timestamp_ms;
    int16_t accel_x;
    int16_t accel_y;
    int16_t accel_z;
    int16_t gyro_x;
    int16_t gyro_y;
    int16_t gyro_z;
} sample_t;

/******************************************************************************
 * Function prototypes
 ******************************************************************************/

static fsp_err_t collect_gesture(sample_t *samples, size_t num_samples);

/******************************************************************************
 * Functions
 ******************************************************************************/

/* Collect gesture data */
static fsp_err_t collect_gesture(sample_t *samples, size_t num_samples)
{
    fsp_err_t err;
    fsp_err_t ret_err = FSP_SUCCESS;

    /* Turn on LED1 to show that we are recording a gesture */
    pin_write(LED1_PIN, BSP_IO_LEVEL_HIGH);

    /* Collect all samples into buffer (fast, no file I/O) */
    for (uint32_t i = 0; i < num_samples; i++)
    {
        /* Record timestamp */
        samples[i].timestamp_ms = i * SAMPLE_PERIOD_MS;

        /* Read accelerometer */
        err = icm20948_read_accel(&samples[i].accel_x,
                                   &samples[i].accel_y,
                                   &samples[i].accel_z);
        if (FSP_SUCCESS != err)
        {
        	ret_err = err;
            samples[i].accel_x = 0;
            samples[i].accel_y = 0;
            samples[i].accel_z = 0;
        }

        /* Read gyroscope */
        err = icm20948_read_gyro(&samples[i].gyro_x,
                                  &samples[i].gyro_y,
                                  &samples[i].gyro_z);
        if (FSP_SUCCESS != err)
        {
        	ret_err = err;
            samples[i].gyro_x = 0;
            samples[i].gyro_y = 0;
            samples[i].gyro_z = 0;
        }

        /* Wait for next sample */
        R_BSP_SoftwareDelay(SAMPLE_PERIOD_MS, BSP_DELAY_UNITS_MILLISECONDS);
    }

    /* Turn off blue LED to show that we are done recording the gesture */
    pin_write(LED1_PIN, BSP_IO_LEVEL_LOW);

    return ret_err;
}

/******************************************************************************
 * Interrupt service routines (ISRs)
 ******************************************************************************/

/* Main I2C callback that dispatches to driver */
void g_i2c_master1_callback(i2c_master_callback_args_t *p_args)
{
    icm20948_i2c_callback(p_args);
}

/******************************************************************************
 * Main entry point
 ******************************************************************************/

void hal_entry(void)
{
    fsp_err_t err;
    float* input_ptr;
    float* output_ptr;
    bool pressed = false;
    sample_t samples[NUM_SAMPLES];
    uint32_t time_us;

    /* Initialize UART */
    TERM_INIT();
    APP_PRINT("\r\nGesture Inference Demo\r\n");

    /* Initialize and start microsecond timer */
	err = init_timer(&g_timer0_ctrl, &g_timer0_cfg);
	if (FSP_SUCCESS != err)
	{
		APP_PRINT("Error: Timer init failed: %d\r\n", err);
		while(1);
	}

	/* Initialize I2C for IMU */
	err = R_IIC_MASTER_Open(&g_i2c_master1_ctrl, &g_i2c_master1_cfg);
	if (FSP_SUCCESS != err)
	{
		APP_PRINT("Error: I2C Open failed: %d\r\n", err);
		while(1);
	}
	APP_PRINT("I2C initialized\r\n");

	/* Initialize IMU */
	err = icm20948_init(&g_i2c_master1,
						ICM20948_ADDR_AD0_HIGH,
						ICM20948_ACCEL_FS_16G,
						ICM20948_GYRO_FS_2000DPS);
	if (FSP_SUCCESS != err)
	{
		APP_PRINT("Error: IMU init failed: %d\r\n", err);
		while (1);
	}
	APP_PRINT("IMU initialized\r\n");

    /* Workaround: Copy config with correct secure/privilege settings */
    rm_ethosu_cfg_t ethosu_cfg = g_rm_ethosu0_cfg;
    ethosu_cfg.secure_enable = 1;
    ethosu_cfg.privilege_enable = 1;

    /* Initialize Ethos core with modified config */
    err = RM_ETHOSU_Open(&g_rm_ethosu0_ctrl, &ethosu_cfg);
    if (FSP_SUCCESS != err) {
        APP_PRINT("  ERROR: RM_ETHOSU_Open failed with code: %d\r\n", err);
        while(1);
    }
    APP_PRINT("Ethos-U NPU initialized\r\n");

    /* Get pointers to model input/output buffers */
    input_ptr = GetModelInputPtr_input();
    output_ptr = GetModelOutputPtr_output_70010();

    /* Tell the user we can start performing inference */
	APP_PRINT("\r\nReady to perform inference!\r\n");
	APP_PRINT("Press SW1 button to capture and classify a gesture\r\n");

	/* Main loop */
	while (1)
	{
		/* See if SW1 button has been pressed (with debounce logic) */
		err = check_button_sw1(&pressed);
		if (FSP_SUCCESS != err) {
			APP_PRINT("Error: Could not check SW1 button: %d\r\n", err);
		}

		/* If pressed, start the collect > inference sequence */
		if (pressed)
		{
			/* Collect data */
			APP_PRINT("Collecting gesture...\r\n");
			err = collect_gesture(samples, NUM_SAMPLES);
			if (FSP_SUCCESS != err)
			{
				APP_PRINT("Error: Could not collect data: %u\r\n", err);
				break;
			}

			/* Preprocess data (standardize) and store in input buffer */
			for (uint32_t i = 0; i < NUM_SAMPLES; i++)
			{
				/* Standardize acceleration channels */
				input_ptr[(NUM_CHANNELS * i) + 0] =
					((float)samples[i].accel_x - STANDARDIZATION_MEANS[0]) /
					STANDARDIZATION_STD_DEVS[0];
				input_ptr[(NUM_CHANNELS * i) + 1] =
					((float)samples[i].accel_y - STANDARDIZATION_MEANS[1]) /
					STANDARDIZATION_STD_DEVS[1];
				input_ptr[(NUM_CHANNELS * i) + 2] =
					((float)samples[i].accel_z - STANDARDIZATION_MEANS[2]) /
					STANDARDIZATION_STD_DEVS[2];

				/* Standardize gyroscope channels */
				input_ptr[(NUM_CHANNELS * i) + 3] =
					((float)samples[i].gyro_x - STANDARDIZATION_MEANS[3]) /
					STANDARDIZATION_STD_DEVS[3];
				input_ptr[(NUM_CHANNELS * i) + 4] =
					((float)samples[i].gyro_y - STANDARDIZATION_MEANS[4]) /
					STANDARDIZATION_STD_DEVS[4];
				input_ptr[(NUM_CHANNELS * i) + 5] =
					((float)samples[i].gyro_z - STANDARDIZATION_MEANS[5]) /
					STANDARDIZATION_STD_DEVS[5];
			}

			/* Run inference */
			time_us = micros();
			RunModel(true);
			time_us = micros() - time_us;

			/* Find predicted class */
			int predicted_class = 0;
			float max_logit = output_ptr[0];
			for (int i = 1; i < NUM_CLASSES; i++) {
				if (output_ptr[i] > max_logit) {
					max_logit = output_ptr[i];
					predicted_class = i;
				}
			}

			/* Print results */
			APP_PRINT("\r\nResults:\r\n");
			APP_PRINT("  Inference time: %u us\r\n", time_us);
			APP_PRINT("  Predicted: %d (%s)\r\n", predicted_class, class_names[predicted_class]);
			APP_PRINT("  Logits:\r\n");
			for (int i = 0; i < NUM_CLASSES; i++) {
				APP_PRINT("    %-15s %12.6f\r\n", class_names[i], output_ptr[i]);
			}
		}
	}
}
