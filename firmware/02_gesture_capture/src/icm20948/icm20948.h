#ifndef ICM20948_H
#define ICM20948_H

/* Standard libraries */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Generated libraries */
#include "hal_data.h"

/* ICM-20948 bank 1 registers */
#define ICM20948_WHO_AM_I           0x00
#define ICM20948_PWR_MGMT_1         0x06
#define ICM20948_PWR_MGMT_2         0x07
#define ICM20948_ACCEL_XOUT_H       0x2D
#define ICM20948_GYRO_XOUT_H        0x33

// Bank 2 registers
#define ICM20948_GYRO_CONFIG_1      0x01
#define ICM20948_ACCEL_CONFIG       0x14

/* ICM-20948 I2C device addresses */
typedef enum e_icm20948_addr
{
    ICM20948_ADDR_AD0_LOW   = 0x68,  /* AD0 pin = LOW (default) */
    ICM20948_ADDR_AD0_HIGH  = 0x69   /* AD0 pin = HIGH */
} icm20948_addr_t;

/* Gyroscope full scale range options */
typedef enum e_icm20948_gyro_fs
{
    ICM20948_GYRO_FS_250DPS  = 0x00,
    ICM20948_GYRO_FS_500DPS  = 0x02,
    ICM20948_GYRO_FS_1000DPS = 0x04,
    ICM20948_GYRO_FS_2000DPS = 0x06
} icm20948_gyro_fs_t;

/* Accelerometer full scale range options */
typedef enum e_icm20948_accel_fs
{
    ICM20948_ACCEL_FS_2G  = 0x00,
    ICM20948_ACCEL_FS_4G  = 0x02,
    ICM20948_ACCEL_FS_8G  = 0x04,
    ICM20948_ACCEL_FS_16G = 0x06
} icm20948_accel_fs_t;

/* I2C timeout */
#define ICM20948_TIMEOUT_MS         1000

/* Function prototypes */
void icm20948_i2c_callback(i2c_master_callback_args_t *p_args);
fsp_err_t icm20948_init(i2c_master_instance_t const * p_i2c_instance,
                        icm20948_addr_t device_addr,
                        icm20948_accel_fs_t accel_range,
                        icm20948_gyro_fs_t gyro_range);
fsp_err_t icm20948_read_accel(int16_t *p_accel_x,
                               int16_t *p_accel_y,
                               int16_t *p_accel_z);
fsp_err_t icm20948_read_gyro(int16_t *p_gyro_x,
                              int16_t *p_gyro_y,
                              int16_t *p_gyro_z);

#endif /* ICM20948_H */
