#include "icm20948.h"

/* User bank selection register */
#define ICM20948_REG_BANK_SEL       0x7F

/* Expected WHO_AM_I value */
#define ICM20948_WHO_AM_I_VAL       0xEA

/* Uninitialized device address */
#define ICM20948_UNINIT_ADDR        0xFFFF

/* Private variables */
static uint32_t g_icm20948_addr = ICM20948_UNINIT_ADDR;
static i2c_master_instance_t const * gp_i2c_instance = NULL;
static volatile bool g_i2c_tx_complete = false;
static volatile bool g_i2c_rx_complete = false;

/* Private function declarations */
static fsp_err_t icm20948_select_bank(uint8_t bank);
static fsp_err_t icm20948_write_reg(uint8_t reg, uint8_t value);
fsp_err_t icm20948_read_regs(uint8_t reg, uint8_t *data, uint8_t len);

/******************************************************************************
 * Private functions
 *****************************************************************************/

/* Select register bank */
static fsp_err_t icm20948_select_bank(uint8_t bank)
{
    return icm20948_write_reg(ICM20948_REG_BANK_SEL, (uint8_t)(bank << 4));
}

/* Write a value to a register */
static fsp_err_t icm20948_write_reg(uint8_t reg, uint8_t value)
{
    fsp_err_t err;
    uint8_t data[2] = {reg, value};

    /* Make sure driver is initialized */
    if (NULL == gp_i2c_instance)
    {
        return FSP_ERR_NOT_INITIALIZED;
    }

    /* Set device address on bus */
    err = R_IIC_MASTER_SlaveAddressSet(gp_i2c_instance->p_ctrl,
                                       g_icm20948_addr,
                                       I2C_MASTER_ADDR_MODE_7BIT);
    if (FSP_SUCCESS != err) {
        return err;
    }

    /* Perform write */
    g_i2c_tx_complete = false;
    err = R_IIC_MASTER_Write(gp_i2c_instance->p_ctrl, data, 2, false);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Wait until we get the transmit complete signal */
    uint32_t timeout = ICM20948_TIMEOUT_MS;
    while (!g_i2c_tx_complete && timeout) {
        R_BSP_SoftwareDelay(1, BSP_DELAY_UNITS_MILLISECONDS);
        timeout--;
    }

    return (g_i2c_tx_complete) ? FSP_SUCCESS : FSP_ERR_TIMEOUT;
}

/* Read [len] number of bytes from a register address */
fsp_err_t icm20948_read_regs(uint8_t reg, uint8_t *data, uint8_t len)
{
    fsp_err_t err;

    /* Make sure driver is initialized */
    if (NULL == gp_i2c_instance)
    {
        return FSP_ERR_NOT_INITIALIZED;
    }

    /* Set device address on bus */
    err = R_IIC_MASTER_SlaveAddressSet(gp_i2c_instance->p_ctrl,
                                       g_icm20948_addr,
                                       I2C_MASTER_ADDR_MODE_7BIT);
    if (FSP_SUCCESS != err) {
        return err;
    }

    /* Perform register address write */
    g_i2c_tx_complete = false;
    err = R_IIC_MASTER_Write(gp_i2c_instance->p_ctrl, &reg, 1, true);
    if (FSP_SUCCESS != err) return err;

    /* Wait until we get the transmit complete signal */
    uint32_t timeout = ICM20948_TIMEOUT_MS;
    while (!g_i2c_tx_complete && timeout--) {
        R_BSP_SoftwareDelay(1, BSP_DELAY_UNITS_MILLISECONDS);
    }
    if (!g_i2c_tx_complete)
    {
        return FSP_ERR_TIMEOUT;
    }

    /* Perform read */
    g_i2c_rx_complete = false;
    err = R_IIC_MASTER_Read(gp_i2c_instance->p_ctrl, data, len, false);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Wait until we get the receive complete signal */
    timeout = ICM20948_TIMEOUT_MS;
    while (!g_i2c_rx_complete && timeout--) {
        R_BSP_SoftwareDelay(1, BSP_DELAY_UNITS_MILLISECONDS);
    }

    return (g_i2c_rx_complete) ? FSP_SUCCESS : FSP_ERR_TIMEOUT;
}

/******************************************************************************
 * Public functions
 *****************************************************************************/

/* Internal I2C callback for ICM-20948 driver */
void icm20948_i2c_callback(i2c_master_callback_args_t *p_args)
{
    if (I2C_MASTER_EVENT_TX_COMPLETE == p_args->event)
    {
        g_i2c_tx_complete = true;
    }
    else if (I2C_MASTER_EVENT_RX_COMPLETE == p_args->event)
    {
        g_i2c_rx_complete = true;
    }
}

/* Initialize the ICM-20948 driver and device */
fsp_err_t icm20948_init(i2c_master_instance_t const * p_i2c_instance,
                        icm20948_addr_t device_addr,
                        icm20948_accel_fs_t accel_range,
                        icm20948_gyro_fs_t gyro_range)
{
    fsp_err_t err;
    uint8_t who_am_i;

    /* Ensure we have a valid I2C handle */
    if (NULL == p_i2c_instance)
    {
        return FSP_ERR_INVALID_POINTER;
    }

    /* Store I2C instance and device address */
    gp_i2c_instance = p_i2c_instance;
    g_icm20948_addr = (uint32_t)device_addr;

    /* Make sure we're in Bank 0 */
    err = icm20948_select_bank(0);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Read WHO_AM_I register */
    err = icm20948_read_regs(ICM20948_WHO_AM_I, &who_am_i, 1);
    if (FSP_SUCCESS != err) {
        return err;
    }

    /* Verify the WHO_AM_I register value */
    if (who_am_i != ICM20948_WHO_AM_I_VAL) {
        return FSP_ERR_NOT_FOUND;
    }

    /* Reset device */
    err = icm20948_write_reg(ICM20948_PWR_MGMT_1, 0x80);
    if (FSP_SUCCESS != err)
    {
        return err;
    }
    R_BSP_SoftwareDelay(100, BSP_DELAY_UNITS_MILLISECONDS);

    /* Wake up device and set clock source */
    err = icm20948_write_reg(ICM20948_PWR_MGMT_1, 0x01);
    if (FSP_SUCCESS != err)
    {
        return err;
    }
    R_BSP_SoftwareDelay(10, BSP_DELAY_UNITS_MILLISECONDS);

    /* Enable accelerometer and gyroscope */
    err = icm20948_write_reg(ICM20948_PWR_MGMT_2, 0x00);
    if (FSP_SUCCESS != err)
    {
        return err;
    }
    R_BSP_SoftwareDelay(10, BSP_DELAY_UNITS_MILLISECONDS);

    /* Switch to Bank 2 to configure sensors */
    err = icm20948_select_bank(2);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Configure accelerometer range */
    err = icm20948_write_reg(ICM20948_ACCEL_CONFIG, (uint8_t)accel_range);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Configure gyroscope range */
    err = icm20948_write_reg(ICM20948_GYRO_CONFIG_1, (uint8_t)gyro_range);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    // Switch back to Bank 0
    err = icm20948_select_bank(0);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    return FSP_SUCCESS;
}

/* Read 3-axis values from accelerometer */
fsp_err_t icm20948_read_accel(int16_t *p_accel_x,
                               int16_t *p_accel_y,
                               int16_t *p_accel_z)
{
    fsp_err_t err;
    uint8_t raw_data[6];

    /* Validate input pointers */
    if ((NULL == p_accel_x) || (NULL == p_accel_y) || (NULL == p_accel_z))
    {
        return FSP_ERR_INVALID_POINTER;
    }

    /* Read 6 bytes of accelerometer data */
    err = icm20948_read_regs(ICM20948_ACCEL_XOUT_H, raw_data, 6);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Convert to signed 16-bit values */
    *p_accel_x = (int16_t)((raw_data[0] << 8) | raw_data[1]);
    *p_accel_y = (int16_t)((raw_data[2] << 8) | raw_data[3]);
    *p_accel_z = (int16_t)((raw_data[4] << 8) | raw_data[5]);

    return FSP_SUCCESS;
}

/* Read 3-axis values from gyroscope */
fsp_err_t icm20948_read_gyro(int16_t *p_gyro_x,
                              int16_t *p_gyro_y,
                              int16_t *p_gyro_z)
{
    fsp_err_t err;
    uint8_t raw_data[6];

    /* Validate input pointers */
    if ((NULL == p_gyro_x) || (NULL == p_gyro_y) || (NULL == p_gyro_z))
    {
        return FSP_ERR_INVALID_POINTER;
    }

    /* Read 6 bytes of gyroscope data */
    err = icm20948_read_regs(ICM20948_GYRO_XOUT_H, raw_data, 6);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Convert to signed 16-bit values */
    *p_gyro_x = (int16_t)((raw_data[0] << 8) | raw_data[1]);
    *p_gyro_y = (int16_t)((raw_data[2] << 8) | raw_data[3]);
    *p_gyro_z = (int16_t)((raw_data[4] << 8) | raw_data[5]);

    return FSP_SUCCESS;
}
