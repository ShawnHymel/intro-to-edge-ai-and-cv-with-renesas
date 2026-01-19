#ifndef UTILS_H
#define UTILS_H

/* Standard libraries */
#include <stdbool.h>
#include <inttypes.h>

/* Generated libraries */
#include "hal_data.h"

/* Custom libraries */
#include "fatfs/ff.h"

/* Settings */
#define DEBOUNCE_MS      40

/* Pins */
#define SW1_PIN         BSP_IO_PORT_00_PIN_09   /* SW1 on P009 */
#define SW2_PIN         BSP_IO_PORT_00_PIN_08   /* SW2 on P008*/
#define LED1_PIN        BSP_IO_PORT_06_PIN_00   /* Blue LED on P600 */
#define LED2_PIN        BSP_IO_PORT_03_PIN_03   /* Green LED on P303 */
#define LED3_PIN        BSP_IO_PORT_10_PIN_07   /* Red LED on PA07 */

/* Function prototypes */
fsp_err_t init_timer(timer_ctrl_t * const timer_ctrl,
                      timer_cfg_t const * const timer_cfg);
uint32_t micros(void);
uint32_t millis(void);
void pin_write(bsp_io_port_pin_t pin, bsp_io_level_t level);
fsp_err_t check_button_sw1(bool *pressed);
fsp_err_t check_button_sw2(bool *pressed);
bool parse_csv_filename(const char *filename, uint32_t *p_num);
bool parse_bmp_filename(const char *filename, uint32_t *p_num);
FRESULT find_next_file_number(uint32_t *file_num);

#endif /* UTILS_H */
