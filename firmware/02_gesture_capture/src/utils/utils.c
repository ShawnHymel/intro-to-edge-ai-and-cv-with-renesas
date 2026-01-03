#include "utils.h"

/* Variables local to module */
static volatile uint32_t s_millis = 0;

/* Must be called from application every 1 ms */
void millis_callback(timer_callback_args_t *p_args)
{
    /* Unused args: suppress warning */
    (void)p_args;

    /* Increment millisecond counter */
    s_millis++;
}

/* Get milliseconds since boot */
uint32_t millis(void)
{
    return s_millis;
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
