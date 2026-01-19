/* Generated libraris */
#include "hal_data.h"

/* Custom libraries */
#include "camera_layer/camera_layer.h"
#include "common_utils.h"
#include "display_layer/display_screen.h"
#include "fatfs/ff.h"
#include "fatfs/diskio.h"
#include "utils/utils.h"

/******************************************************************************
 * Settings
 ******************************************************************************/

/* BMP output resolution (cropped and scaled from camera input) */
#define BMP_SAVE_WIDTH   352
#define BMP_SAVE_HEIGHT  352

/* Adjust color/brightness of saved BMP (per-channel) */
/* 100 = no change, 80 = 20% darker, 120 = 20% brighter */
#define BMP_RED_PERCENT    170
#define BMP_GREEN_PERCENT  150
#define BMP_BLUE_PERCENT   200

/* BMP write buffer settings */
#define BMP_ROWS_PER_WRITE  8

/* Row bytes padded to 4-byte boundary */
#define BMP_ROW_BYTES       (((BMP_SAVE_WIDTH * 3) + 3) & ~3)

/******************************************************************************
 * Static variables
 ******************************************************************************/

static uint8_t bmp_write_buffer[BMP_ROWS_PER_WRITE * BMP_ROW_BYTES];

/******************************************************************************
 * Function prototypes
 ******************************************************************************/

static FRESULT save_image_as_bmp(uint32_t file_num, uint8_t *rgb565_buffer,
                                 uint16_t src_width, uint16_t src_height,
                                 uint16_t out_width, uint16_t out_height);

/******************************************************************************
 * Functions
 ******************************************************************************/

/* Given a file number, save image to BMP file in the root of the SD card */
static FRESULT save_image_as_bmp(uint32_t file_num, uint8_t *rgb565_buffer,
                                 uint16_t src_width, uint16_t src_height,
                                 uint16_t out_width, uint16_t out_height)
{
    FRESULT fr;
    FIL file;
    UINT bytes_written;
    char filename[13];

    /* Generate filename: XXXX.BMP */
    sprintf(filename, "%04lu.BMP", (unsigned long)file_num);

    /* Calculate BMP parameters */
    uint32_t row_size = ((out_width * 3 + 3) / 4) * 4;  /* Row size padded to 4 bytes */
    uint32_t pixel_data_size = row_size * out_height;
    uint32_t file_size = 54 + pixel_data_size;  /* 54 byte header + pixel data */

    /* BMP file header (14 bytes) */
    uint8_t bmp_header[54] = {
        /* File header (14 bytes) */
        'B', 'M',                                           /* Signature */
        (uint8_t)(file_size), (uint8_t)(file_size >> 8),    /* File size (little endian) */
        (uint8_t)(file_size >> 16), (uint8_t)(file_size >> 24),
        0, 0, 0, 0,                                         /* Reserved */
        54, 0, 0, 0,                                        /* Pixel data offset */

        /* DIB header (40 bytes - BITMAPINFOHEADER) */
        40, 0, 0, 0,                                        /* DIB header size */
        (uint8_t)(out_width), (uint8_t)(out_width >> 8),    /* Width */
        (uint8_t)(out_width >> 16), (uint8_t)(out_width >> 24),
        (uint8_t)(out_height), (uint8_t)(out_height >> 8),  /* Height */
        (uint8_t)(out_height >> 16), (uint8_t)(out_height >> 24),
        1, 0,                                               /* Color planes */
        24, 0,                                              /* Bits per pixel */
        0, 0, 0, 0,                                         /* Compression (none) */
        (uint8_t)(pixel_data_size), (uint8_t)(pixel_data_size >> 8),  /* Image size */
        (uint8_t)(pixel_data_size >> 16), (uint8_t)(pixel_data_size >> 24),
        0, 0, 0, 0,                                         /* X pixels per meter */
        0, 0, 0, 0,                                         /* Y pixels per meter */
        0, 0, 0, 0,                                         /* Colors in color table */
        0, 0, 0, 0                                          /* Important colors */
    };

    /* Calculate crop parameters (center crop to square) */
    uint16_t crop_size = (src_width < src_height) ? src_width : src_height;
    uint16_t crop_x = (src_width - crop_size) / 2;
    uint16_t crop_y = (src_height - crop_size) / 2;

    /* Open file for writing */
    fr = f_open(&file, filename, FA_CREATE_ALWAYS | FA_WRITE);
    if (FR_OK != fr)
    {
        return fr;
    }

    /* Write BMP header */
    fr = f_write(&file, bmp_header, 54, &bytes_written);
    if ((FR_OK != fr) || (54 != bytes_written))
    {
        f_close(&file);
        return (FR_OK != fr) ? fr : FR_DISK_ERR;
    }

    /* Write pixel data (BMP stores rows bottom-to-top) */
    /* Process in chunks of BMP_ROWS_PER_WRITE rows */
    uint16_t rows_in_buffer = 0;

    for (int32_t out_y = out_height - 1; out_y >= 0; out_y--)
    {
        /* Calculate source Y with scaling */
        uint16_t src_y = crop_y + (uint16_t)(((uint32_t)out_y * crop_size) / out_height);

        /* Pointer to current row in buffer */
        uint8_t *row_ptr = &bmp_write_buffer[rows_in_buffer * row_size];

        for (uint16_t out_x = 0; out_x < out_width; out_x++)
        {
            /* Calculate source X with scaling */
            uint16_t src_x = crop_x + (uint16_t)(((uint32_t)out_x * crop_size) / out_width);

            /* Get RGB565 pixel (2 bytes per pixel) */
            uint32_t src_offset = (src_y * src_width + src_x) * 2;
            uint16_t rgb565 = (uint16_t)(rgb565_buffer[src_offset] |
                                         (rgb565_buffer[src_offset + 1] << 8));

            /* Convert RGB565 to BGR888 */
            uint16_t r = (uint16_t)(((rgb565 >> 11) & 0x1F) << 3);
            uint16_t g = (uint16_t)(((rgb565 >> 5) & 0x3F) << 2);
            uint16_t b = (uint16_t)((rgb565 & 0x1F) << 3);

            /* Apply per-channel color/brightness adjustment and clamp to 255 */
            r = (uint16_t)((r * BMP_RED_PERCENT) / 100);   if (r > 255) r = 255;
            g = (uint16_t)((g * BMP_GREEN_PERCENT) / 100); if (g > 255) g = 255;
            b = (uint16_t)((b * BMP_BLUE_PERCENT) / 100);  if (b > 255) b = 255;

            /* Store as BGR (BMP format) */
            row_ptr[out_x * 3 + 0] = (uint8_t)b;
            row_ptr[out_x * 3 + 1] = (uint8_t)g;
            row_ptr[out_x * 3 + 2] = (uint8_t)r;
        }

        rows_in_buffer++;

        /* Write buffer when full or on last row */
        if ((rows_in_buffer == BMP_ROWS_PER_WRITE) || (out_y == 0))
        {
            uint32_t bytes_to_write = rows_in_buffer * row_size;
            fr = f_write(&file, bmp_write_buffer, bytes_to_write, &bytes_written);
            if ((FR_OK != fr) || (bytes_to_write != bytes_written))
            {
                f_close(&file);
                return (FR_OK != fr) ? fr : FR_DISK_ERR;
            }
            rows_in_buffer = 0;
        }
    }

    /* Close file */
    fr = f_close(&file);

    return fr;
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

    /* Initialize debugging terminal */
    TERM_INIT();
    APP_PRINT("Image capture\r\n");

    /* Initialize and start microsecond timer */
    err = init_timer(&g_timer0_ctrl, &g_timer0_cfg);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Timer init failed: %d\r\n", err);
        while(1);
    }

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
    APP_PRINT("Press SW1 button to capture an image \r\n");

    /* Enable MIPI I/F on the EK-RA8P1 */
    err = R_IOPORT_PinWrite(&g_ioport_ctrl, MIPI_IF_EN, BSP_IO_LEVEL_LOW);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not toggle MIPI_IF_EN pin\r\n");
        while (1);
    }

    /* Initialize display */
    APP_PRINT("Initializing display...\r\n");
    display_image_buffer_initialize();

    /* Initialize the 2D draw engine */
    err = drw_init();
    if(FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not initialize 2D draw enginer\r\n");
        while (1);
    }

    /* Initialize the display peripheral module and connected LCD display */
    err = display_init();
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not initialize display\r\n");
        while (1);
    }

    /* Clear camera image buffer */
    APP_PRINT("Initializing camera...\r\n");
    camera_image_buffer_initialize();

    /* Initialize the camera capture peripheral module and connected camera */
    err = camera_init(false);
    if (FSP_SUCCESS != err)
    {
        APP_PRINT("Error: Could not initialize camera\r\n");
    }

    /* Start camera capture */
    APP_PRINT("Starting camera capture...\r\n");
    camera_capture_start();

    /* Main loop */
    bool save_pending = false;
    while (1)
    {
        /* See if SW1 button has been pressed (with debounce logic) */
        err = check_button_sw1(&pressed);
        if (FSP_SUCCESS != err) {
            APP_PRINT("Error: Could not check SW1 button: %d\r\n", err);
        }

        /* If pressed, set flag to save on next frame */
        if (pressed)
        {
            save_pending = true;
        }

        /* Poll for camera ready flag */
        if (g_camera_frame_ready)
        {
            g_camera_frame_ready = false;

            /* Post processing for camera image capture. After this process is completed,
             * user app can take an image from camera_capture_image_rgb565[].
             */
            camera_capture_post_process();

            /* If save is pending, save the image now that the buffer is valid */
            if (save_pending)
            {
                save_pending = false;

                /* Turn on red LED to indicate SD card access */
                pin_write(LED3_PIN, BSP_IO_LEVEL_HIGH);

                APP_PRINT("Saving image %04lu.BMP...\r\n", (unsigned long)file_num);

                /* Invalidate cache to ensure CPU sees latest data from DMA */
                SCB_InvalidateDCache_by_Addr((uint32_t *)camera_capture_image_rgb565,
                                             (int32_t)camera_capture_image_rgb565_size);

                /* Save the current camera frame as BMP (crop to square, scale to output size) */
                fr = save_image_as_bmp(file_num,
                                       camera_capture_image_rgb565,
                                       CAMERA_CAPTURE_IMAGE_WIDTH,
                                       CAMERA_CAPTURE_IMAGE_HEIGHT,
                                       BMP_SAVE_WIDTH,
                                       BMP_SAVE_HEIGHT);
                if (FR_OK == fr)
                {
                    APP_PRINT("Saved %04lu.BMP successfully\r\n", (unsigned long)file_num);
                    file_num++;
                }
                else
                {
                    APP_PRINT("Error: Failed to save BMP: %d\r\n", fr);
                }

                /* Turn off red LED */
                pin_write(LED3_PIN, BSP_IO_LEVEL_LOW);
            }

            /* Wait for vsync flag */
            while (!g_display_vsync_ready);
            g_display_vsync_ready = false;

            /* Start a new graphics frame */
            graphics_start_frame();

            /* Display the camera image */
            display_camera_image();

            /* Wait for previous frame rendering to finish, then finalize this frame and flip the buffers */
            graphics_end_frame();
        }
    }
}
