#include <stdlib.h>
#include <stdio.h>

#include "utils.h"

/* Store the timer handle */
static timer_ctrl_t * sp_timer_ctrl = NULL;

/* Initialize and start timer */
fsp_err_t init_timer(timer_ctrl_t * const timer_ctrl,
                     timer_cfg_t const * const timer_cfg)
{
    fsp_err_t err;

    /* Initialize timer and apply the configuration settings */
    err = R_GPT_Open(timer_ctrl, timer_cfg);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Start the timer */
    err = R_GPT_Start(timer_ctrl);
    if (FSP_SUCCESS != err)
    {
        /* Start failed: close the timer and throw error */
        R_GPT_Close(timer_ctrl);
        return err;
    }

    /* Success: store pointer to control struct */
    sp_timer_ctrl = timer_ctrl;

    return err;
}

/* Get microseconds since boot */
uint32_t micros(void)
{
    fsp_err_t err;
    timer_status_t status;

    /* Early return if timer is not initialized */
    if (NULL == sp_timer_ctrl)
    {
        return 0;
    }

    /* Get timer count */
    err = R_GPT_StatusGet(sp_timer_ctrl, &status);
    if (FSP_SUCCESS != err)
    {
        return 0;
    }

    return status.counter;
}

/* Get milliseconds since boot */
uint32_t millis(void)
{
    return micros() / 1000;
}

/* Wrapper for setting an output pin to high or low */
void pin_write(bsp_io_port_pin_t pin, bsp_io_level_t level)
{
    R_BSP_PinAccessEnable();
    R_BSP_PinWrite(pin, level);
    R_BSP_PinAccessDisable();
}


/* Non-blocking debounce button check for SW1. Call often in main loop. */
fsp_err_t check_button_sw1(bool *pressed)
{
    static bsp_io_level_t button_state = BSP_IO_LEVEL_HIGH;
    static bsp_io_level_t last_button_state = BSP_IO_LEVEL_HIGH;
    static uint32_t last_debounce_time = 0;
    fsp_err_t err;
    bsp_io_level_t reading;

    /* Initialize pressed */
    *pressed = false;

    /* Get button state */
    err = R_IOPORT_PinRead(&g_ioport_ctrl, SW1_PIN, &reading);
    if (FSP_SUCCESS != err)
    {
        return err;
    }

    /* Check if the switch changed state */
    if (reading != last_button_state)
    {
        last_debounce_time = millis();
    }

    /* Save current reading for next iteration */
    last_button_state = reading;

    /* Check if reading has been stable for debounce period */
    if ((millis() - last_debounce_time) > DEBOUNCE_MS)
    {
        /* If the button state has changed */
        if (reading != button_state)
        {
            button_state = reading;

            /* Detect falling edge (button is active low) */
            if (BSP_IO_LEVEL_LOW == button_state)
            {
                *pressed = true;
            }
        }
    }

    return FSP_SUCCESS;
}

/* Non-blocking debounce button check for SW2. Call often in main loop. */
fsp_err_t check_button_sw2(bool *pressed)
{
    static bsp_io_level_t button_state = BSP_IO_LEVEL_HIGH;
        static bsp_io_level_t last_button_state = BSP_IO_LEVEL_HIGH;
        static uint32_t last_debounce_time = 0;
        fsp_err_t err;
        bsp_io_level_t reading;

        /* Initialize pressed */
        *pressed = false;

        /* Get button state */
        err = R_IOPORT_PinRead(&g_ioport_ctrl, SW2_PIN, &reading);
        if (FSP_SUCCESS != err)
        {
            return err;
        }

        /* Check if the switch changed state */
        if (reading != last_button_state)
        {
            last_debounce_time = millis();
        }

        /* Save current reading for next iteration */
        last_button_state = reading;

        /* Check if reading has been stable for debounce period */
        if ((millis() - last_debounce_time) > DEBOUNCE_MS)
        {
            /* If the button state has changed */
            if (reading != button_state)
            {
                button_state = reading;

                /* Detect falling edge (button is active low) */
                if (BSP_IO_LEVEL_LOW == button_state)
                {
                    *pressed = true;
                }
            }
        }

        return FSP_SUCCESS;
}
