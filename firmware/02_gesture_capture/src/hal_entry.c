/* Standard libraries */
#include <inttypes.h>

/* Generated code */
#include "hal_data.h"

/* Custom libraries */
#include "common_utils.h"
#include "icm20948/icm20948.h"
#include "fatfs/ff.h"
#include "fatfs/diskio.h"
#include "utils/utils.h"

/******************************************************************************
 * Settings
 ******************************************************************************/

/* Data collection */
#define SAMPLE_RATE_HZ      100
#define SAMPLE_PERIOD_MS    10
#define NUM_SAMPLES         100 /* 1 sec */

/******************************************************************************
 * Data Structures
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
 * Global Variables
 ******************************************************************************/


/******************************************************************************
 * Function Prototypes
 ******************************************************************************/

static bool parse_csv_filename(const char *filename, uint32_t *p_num);
static FRESULT find_next_file_number(uint32_t *file_num);
static FRESULT collect_gesture(uint32_t file_num);

/******************************************************************************
 * Functions
 ******************************************************************************/

/* Check if filename matches XXXXXX.CSV pattern and extract number */
static bool parse_csv_filename(const char *filename, uint32_t *p_num)
{
    /* Check length: should be exactly "XXXXXX.CSV" (10 chars) */
    if (10 != strlen(filename))
    {
        return false;
    }

    /* Check extension */
    if (0 != strcmp(&filename[7], "CSV"))
    {
        return false;
    }

    /* Check if first 6 chars are digits */
    for (int i = 0; i < 6; i++)
    {
        if (('0' > filename[i]) || ('9' < filename[i]))
        {
            return false;
        }
    }

    /* Check dot */
    if ('.' != filename[6])
    {
        return false;
    }

    /* Extract number */
    char num_str[7];
    memcpy(num_str, filename, 6);
    num_str[6] = '\0';
    *p_num = (uint32_t)atol(num_str);

    return true;
}

/* Scan SD card and find next file number */
static FRESULT find_next_file_number(uint32_t *file_num)
{
    FRESULT fr;
    DIR dir;
    FILINFO fno;
    uint32_t max_num = 0;
    bool found = false;

    APP_PRINT("Scanning SD card for existing files...\r\n");

    /* Open root directory */
    fr = f_opendir(&dir, "/");
    if (FR_OK != fr) {
        return fr;
    }

    /* Scan all files */
    while (1) {
        /* Read directory entries */
        fr = f_readdir(&dir, &fno);
        if ((FR_OK != fr) || (0 == fno.fname[0]))
        {
            break;
        }

        /* Skip directories */
        if (fno.fattrib & AM_DIR)
        {
            continue;
        }

        /* Check if filename matches our pattern */
        if (parse_csv_filename(fno.fname, file_num))
        {
            found = true;
            if (*file_num > max_num)
            {
                max_num = *file_num;
            }
        }
    }

    /* Close directory */
    fr = f_closedir(&dir);
    if (FR_OK != fr)
    {
        return fr;
    }

    /* Set next file number */
    if (found)
    {
        *file_num = max_num + 1;
    }
    else
    {
        *file_num = 0;
    }

    return FR_OK;
}

/* Collect gesture data and save to SD card */
static FRESULT collect_gesture(uint32_t file_num)
{
    fsp_err_t err;
    FRESULT fr;
    FIL fil;
    UINT bw;
    char filename[16];
    sample_t samples[NUM_SAMPLES];

    /* Turn on LED1 to show that we are recording a gesture */
    pin_write(LED1_PIN, BSP_IO_LEVEL_HIGH);

    /* Collect all samples into buffer (fast, no file I/O) */
    for (uint32_t i = 0; i < NUM_SAMPLES; i++)
    {
        /* Record timestamp */
        samples[i].timestamp_ms = i * SAMPLE_PERIOD_MS;

        /* Read accelerometer */
        err = icm20948_read_accel(&samples[i].accel_x,
                                   &samples[i].accel_y,
                                   &samples[i].accel_z);
        if (FSP_SUCCESS != err)
        {
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
            samples[i].gyro_x = 0;
            samples[i].gyro_y = 0;
            samples[i].gyro_z = 0;
        }

        /* Wait for next sample */
        R_BSP_SoftwareDelay(SAMPLE_PERIOD_MS, BSP_DELAY_UNITS_MILLISECONDS);
    }
    APP_PRINT("Data collection complete. Writing to SD card...\r\n");

    /* Turn off blue LED to show that we are done recording the gesture */
    pin_write(LED1_PIN, BSP_IO_LEVEL_LOW);

    /* Turn on red LED to show that we are accessing the SD card */
    pin_write(LED3_PIN, BSP_IO_LEVEL_HIGH);

    /* Create filename */
    snprintf(filename, sizeof(filename), "%06" PRIu32 ".CSV", file_num);

    /* Create CSV file */
    fr = f_open(&fil, filename, FA_CREATE_ALWAYS | FA_WRITE);
    if (fr != FR_OK)
    {
        APP_PRINT("Error: File open failed: %d\r\n", fr);
        return fr;
    }
    APP_PRINT("File created: %s\r\n", filename);

    /* Write CSV header */
    const char *header = "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y," \
                         "gyro_z\r\n";
    fr = f_write(&fil, header, strlen(header), &bw);
    if (fr != FR_OK)
    {
        APP_PRINT("Error: Header write failed: %d\r\n", fr);
        f_close(&fil);
        return fr;
    }

    /* Write all buffered samples */
    for (uint32_t i = 0; i < NUM_SAMPLES; i++)
    {
        char line[100];
        int len = snprintf(line,
                           sizeof(line),
                           "%" PRIu32 ",%d,%d,%d,%d,%d,%d\r\n",
                           samples[i].timestamp_ms,
                           samples[i].accel_x,
                           samples[i].accel_y,
                           samples[i].accel_z,
                           samples[i].gyro_x,
                           samples[i].gyro_y,
                           samples[i].gyro_z);

        fr = f_write(&fil, line, (UINT)len, &bw);
        if ((FR_OK != fr) || (bw != (UINT)len))
        {
            APP_PRINT("Data write failed at sample %lu\r\n", i);
            break;
        }
    }

    /* Close file */
    fr = f_close(&fil);
    if (fr != FR_OK) {
        APP_PRINT("Error: File close failed: %d\r\n", fr);
        return fr;
    }
    APP_PRINT("Recorded %u samples\r\n", NUM_SAMPLES);

    /* Turn off red LED to show that we are done accessing the SD card */
    pin_write(LED3_PIN, BSP_IO_LEVEL_LOW);

   return FR_OK;
}

/******************************************************************************
 * Interrupt service routines (ISRs)
 ******************************************************************************/

/* Timer 0 callback: dispatch to millis handler */
void g_timer0_callback(timer_callback_args_t *p_args)
{
    millis_callback(p_args);
}

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
    FRESULT fr;
    FATFS fs;
    uint32_t file_num = 0;
    bool pressed = false;

    /* Initialize UART */
    TERM_INIT();
    APP_PRINT("\r\nGesture Sample Collection\r\n");

    /* Initialize millisecond timer */
    err = R_GPT_Open(&g_timer0_ctrl, &g_timer0_cfg);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Timer Open failed: %d\r\n", err);
        while(1);
    }

    /* Start millisecond timer */
    err = R_GPT_Start(&g_timer0_ctrl);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Timer Start failed: %d\r\n", err);
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

    /* Initialize SPI for SD card */
    err = R_SCI_B_SPI_Open(&g_sci_spi0_ctrl, &g_sci_spi0_cfg);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: SPI Open failed: %d\r\n", err);
        while (1);
    }
    APP_PRINT("SPI initialized\r\n");

    /* Turn on red LED to show that we are accessing the SD card */
    pin_write(LED3_PIN, BSP_IO_LEVEL_HIGH);

    /* Mount SD card */
    fr = f_mount(&fs, "", 1);
    if (FR_OK != fr) {
        APP_PRINT("Error: SD mount failed: %d\r\n", fr);
        while (1);
    }
    APP_PRINT("SD card mounted\r\n");

    /* Find next file number */
    fr = find_next_file_number(&file_num);
    if (FR_OK != fr)
    {
        APP_PRINT("Error: Failed to open directory: %d\r\n", fr);
        while (1);
    }
    APP_PRINT("Next file number: %" PRIu32 "\r\n", file_num);

    /* Turn off red LED to show that we are done accessing the SD card */
    pin_write(LED3_PIN, BSP_IO_LEVEL_LOW);

    /* Tell the user we can start collecting samples */
    APP_PRINT("\r\nReady for data collection!\r\n");
    APP_PRINT("Press SW1 button to capture a gesture \r\n");
    APP_PRINT("Each capture: %d readings at %d Hz\r\n",
              NUM_SAMPLES,
              SAMPLE_RATE_HZ);

    /* Main loop */
    while (1)
    {
        /* See if SW1 button has been pressed (with debounce logic) */
        err = check_button_sw1(&pressed);
        if (FSP_SUCCESS != err) {
            APP_PRINT("Error: Could not check SW1 button: %d\r\n", err);
        }

        /* If pressed, collect data (blocking) */
        if (pressed)
        {
            APP_PRINT("Collecting data...\r\n");
            fr = collect_gesture(file_num);
            if (FR_OK != fr)
            {
                APP_PRINT("Error: Could not collect data\r\n");
                break;
            }

            /* Increment file number */
            file_num++;
        }
    }
}
