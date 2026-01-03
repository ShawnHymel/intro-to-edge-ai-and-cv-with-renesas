/**
 * Low level disk I/O module for FatFs on Renesas RA (SPI SD Card)
 */

#include "ff.h"
#include "diskio.h"
#include "hal_data.h"

/*-----------------------------------------------------------------------*/
/* SD Card SPI Configuration                                             */
/*-----------------------------------------------------------------------*/

/* External SPI instance (from FSP configuration) */
extern const spi_instance_t g_sci_spi0;

/* CS pin definition */
#define SD_CS_PIN    BSP_IO_PORT_06_PIN_04

/* SD card commands */
#define CMD0    (0)         /* GO_IDLE_STATE */
#define CMD1    (1)         /* SEND_OP_COND (MMC) */
#define CMD8    (8)         /* SEND_IF_COND */
#define CMD9    (9)         /* SEND_CSD */
#define CMD10   (10)        /* SEND_CID */
#define CMD12   (12)        /* STOP_TRANSMISSION */
#define CMD16   (16)        /* SET_BLOCKLEN */
#define CMD17   (17)        /* READ_SINGLE_BLOCK */
#define CMD18   (18)        /* READ_MULTIPLE_BLOCK */
#define ACMD23  (0x80 | 23) /* SET_WR_BLK_ERASE_COUNT (for SD) */
#define CMD24   (24)        /* WRITE_BLOCK */
#define CMD25   (25)        /* WRITE_MULTIPLE_BLOCK */
#define CMD55   (55)        /* APP_CMD */
#define CMD58   (58)        /* READ_OCR */
#define ACMD41  (0x80 | 41) /* SEND_OP_COND (ACMD) */

/* Card type flags */
#define CT_MMC      0x01    /* MMC ver 3 */
#define CT_SD1      0x02    /* SD ver 1 */
#define CT_SD2      0x04    /* SD ver 2 */
#define CT_BLOCK    0x08    /* Block addressing */

/*-----------------------------------------------------------------------*/
/* Private Variables                                                     */
/*-----------------------------------------------------------------------*/

static volatile DSTATUS Stat = STA_NOINIT;  /* Disk status */
static BYTE CardType;                        /* Card type flags */
static volatile bool g_spi_complete = false; /* SPI transfer complete flag */

/*-----------------------------------------------------------------------*/
/* SPI Helper Functions                                                  */
/*-----------------------------------------------------------------------*/

/* SPI callback */
void sci_b_spi0_callback(spi_callback_args_t *p_args)
{
    if (SPI_EVENT_TRANSFER_COMPLETE == p_args->event)
    {
        g_spi_complete = true;
    }
}

/* Assert CS (select SD card) */
static inline void SELECT(void)
{
    R_IOPORT_PinWrite(&g_ioport_ctrl, SD_CS_PIN, BSP_IO_LEVEL_LOW);
    R_BSP_SoftwareDelay(1, BSP_DELAY_UNITS_MICROSECONDS);
}

/* Deassert CS (deselect SD card) */
static inline void DESELECT(void)
{
    R_IOPORT_PinWrite(&g_ioport_ctrl, SD_CS_PIN, BSP_IO_LEVEL_HIGH);
    R_BSP_SoftwareDelay(1, BSP_DELAY_UNITS_MICROSECONDS);
}

/* SPI transfer */
static void spi_transfer(const BYTE *tx, BYTE *rx, UINT len)
{
    g_spi_complete = false;
    R_SCI_B_SPI_WriteRead(g_sci_spi0.p_ctrl, tx, rx, len, SPI_BIT_WIDTH_8_BITS);
    while (!g_spi_complete);
}

/* Send single byte */
static BYTE spi_txrx(BYTE data)
{
    BYTE rx;
    spi_transfer(&data, &rx, 1);
    return rx;
}

/* Wait for card ready */
static BYTE wait_ready(UINT timeout_ms)
{
    BYTE res;
    UINT timer = timeout_ms;

    do {
        res = spi_txrx(0xFF);
        if (res == 0xFF) return 1;
        R_BSP_SoftwareDelay(1, BSP_DELAY_UNITS_MILLISECONDS);
    } while (--timer);

    return 0;
}

/* Send command packet */
static BYTE send_cmd(BYTE cmd, DWORD arg)
{
    BYTE n, res;
    BYTE buf[6];

    /* Handle ACMD<n> - special two-step process */
    if (cmd & 0x80) {
        cmd &= 0x7F;

        /* First, send CMD55 (without deselect/reselect) */
        SELECT();  // Make sure CS is low

        /* Build CMD55 packet */
        buf[0] = 0x40 | CMD55;
        buf[1] = 0;
        buf[2] = 0;
        buf[3] = 0;
        buf[4] = 0;
        buf[5] = 0x01;

        /* Send CMD55 */
        spi_transfer(buf, buf, 6);

        /* Get CMD55 response */
        n = 10;
        do {
            res = spi_txrx(0xFF);
        } while ((res & 0x80) && --n);
        if (res > 1) {
            DESELECT();
            return res;  /* CMD55 failed */
        }

        /* CMD55 succeeded - now send the actual command immediately */

    } else {
        /* Regular command - deselect/select cycle */
        DESELECT();
        R_BSP_SoftwareDelay(10, BSP_DELAY_UNITS_MICROSECONDS);
        SELECT();
    }

    /* Build command packet */
    buf[0] = 0x40 | cmd;
    buf[1] = (BYTE)(arg >> 24);
    buf[2] = (BYTE)(arg >> 16);
    buf[3] = (BYTE)(arg >> 8);
    buf[4] = (BYTE)arg;

    /* CRC */
    if (cmd == CMD0) buf[5] = 0x95;
    else if (cmd == CMD8) buf[5] = 0x87;
    else buf[5] = 0x01;

    /* Send command */
    spi_transfer(buf, buf, 6);

    /* Get response */
    n = 10;
    do {
        res = spi_txrx(0xFF);
    } while ((res & 0x80) && --n);

    /* CS stays LOW - caller handles additional data and deselect */

    return res;
}

/*-----------------------------------------------------------------------*/
/* FatFs Disk I/O Functions                                              */
/*-----------------------------------------------------------------------*/

/* Initialize Disk Drive */
DSTATUS disk_initialize(BYTE pdrv)
{
    BYTE n, cmd, ty, ocr[4];
    UINT tmr;
    BYTE res;

    if (pdrv) return STA_NOINIT;  /* Only drive 0 */

    /* Power up */
    for (n = 10; n; n--) spi_txrx(0xFF);

    /* Enter Idle state */
    ty = 0;
    if (send_cmd(CMD0, 0) == 1) {
        tmr = 1000;  /* Initialization timeout = 1 sec */

        if (send_cmd(CMD8, 0x1AA) == 1) {  /* SDv2? */
            for (n = 0; n < 4; n++) ocr[n] = spi_txrx(0xFF);
            DESELECT();

            if (ocr[2] == 0x01 && ocr[3] == 0xAA) {  /* Card supports 2.7-3.6V? */
                /* Wait for leaving idle state (ACMD41 with HCS bit) */
                while (tmr-- && (res = send_cmd(ACMD41, 1UL << 30))) {
                    DESELECT();
                    R_BSP_SoftwareDelay(10, BSP_DELAY_UNITS_MILLISECONDS);
                }
                DESELECT();

                /* Check CCS bit in OCR */
                if (tmr && send_cmd(CMD58, 0) == 0) {
                    for (n = 0; n < 4; n++) ocr[n] = spi_txrx(0xFF);
                    DESELECT();
                    ty = (ocr[0] & 0x40) ? CT_SD2 | CT_BLOCK : CT_SD2;
                }
            }
        } else {  /* SDv1 or MMCv3 */
            if (send_cmd(ACMD41, 0) <= 1) {
                ty = CT_SD1; cmd = ACMD41;  /* SDv1 */
            } else {
                ty = CT_MMC; cmd = CMD1;  /* MMCv3 */
            }

            /* Wait for leaving idle state */
            while (tmr-- && send_cmd(cmd, 0));

            /* Set block length to 512 */
            if (!tmr || send_cmd(CMD16, 512) != 0) ty = 0;
        }
    }

    CardType = ty;
    DESELECT();

    if (ty) {
        Stat &= ~STA_NOINIT;  /* Clear STA_NOINIT */
    } else {
        Stat = STA_NOINIT;
    }

    return Stat;
}

/* Get Disk Status */
DSTATUS disk_status(BYTE pdrv)
{
    if (pdrv) return STA_NOINIT;
    return Stat;
}

/* Read Sector(s) */
DRESULT disk_read(BYTE pdrv, BYTE *buff, LBA_t sector, UINT count)
{
    BYTE cmd;
    BYTE token;
    UINT tmr;

    if (pdrv || !count) return RES_PARERR;
    if (Stat & STA_NOINIT) return RES_NOTRDY;

    /* Convert LBA to byte address if needed */
    if (!(CardType & CT_BLOCK)) sector *= 512;

    /* Perform read action */
    cmd = count > 1 ? CMD18 : CMD17;
    if (send_cmd(cmd, sector) == 0) {
        do {
            if (!wait_ready(500)) {
                break;
            }

            /* Wait for data token */
            tmr = 2000;
            do {
                token = spi_txrx(0xFF);
            } while (token == 0xFF && --tmr);

            /* Not a data token */
            if (token != 0xFE) break;

            /* Read data */
            static BYTE tx_dummy[512];
            if (tx_dummy[0] != 0xFF) {
                memset(tx_dummy, 0xFF, 512);
            }
            spi_transfer(tx_dummy, buff, 512);
            buff += 512;

            /* Discard CRC */
            spi_txrx(0xFF);
            spi_txrx(0xFF);

        } while (--count);

        /* Stop transmission */
        if (cmd == CMD18) send_cmd(CMD12, 0);
    }

    DESELECT();

    return count ? RES_ERROR : RES_OK;
}

/* Write Sector(s) */
DRESULT disk_write(BYTE pdrv, const BYTE *buff, LBA_t sector, UINT count)
{
    if (pdrv || !count) return RES_PARERR;
    if (Stat & STA_NOINIT) return RES_NOTRDY;
    if (Stat & STA_PROTECT) return RES_WRPRT;

    /* Convert LBA to byte address if needed */
    if (!(CardType & CT_BLOCK)) sector *= 512;

    if (count == 1) {  /* Single block write */
        if ((send_cmd(CMD24, sector) == 0) && wait_ready(500)) {
            /* Send data token */
            spi_txrx(0xFE);

            /* Send data (with throwaway rx buffer) */
            static BYTE rx_dummy[512];
            spi_transfer(buff, rx_dummy, 512);

            /* Dummy CRC */
            spi_txrx(0xFF);
            spi_txrx(0xFF);

            /* Check response */
            BYTE res = spi_txrx(0xFF);
            if ((res & 0x1F) == 0x05) {
                wait_ready(500);
                count = 0;
            }
        }
    }

    DESELECT();

    return count ? RES_ERROR : RES_OK;
}

/* Miscellaneous Functions */
DRESULT disk_ioctl(BYTE pdrv, BYTE cmd, void *buff)
{
    DRESULT res = RES_ERROR;
    BYTE n, csd[16];
    DWORD cs;

    if (pdrv) return RES_PARERR;
    if (Stat & STA_NOINIT) return RES_NOTRDY;

    switch (cmd) {
        case CTRL_SYNC:
            SELECT();
            if (wait_ready(500)) res = RES_OK;
            DESELECT();
            break;

        case GET_SECTOR_COUNT:
            if ((send_cmd(CMD9, 0) == 0) && wait_ready(500)) {
                /* Wait for data token */
                UINT tmr = 2000;
                do {
                    n = spi_txrx(0xFF);
                } while (n == 0xFF && --tmr);

                if (n == 0xFE) {
                    for (n = 0; n < 16; n++) csd[n] = spi_txrx(0xFF);
                    spi_txrx(0xFF); spi_txrx(0xFF);  /* CRC */

                    if ((csd[0] >> 6) == 1) {  /* SDC ver 2.00 */
                        cs = csd[9] + ((WORD)csd[8] << 8) + ((DWORD)(csd[7] & 63) << 16) + 1;
                        *(LBA_t*)buff = cs << 10;
                    } else {  /* SDC ver 1.XX or MMC */
                        n = (csd[5] & 15) + ((csd[10] & 128) >> 7) + ((csd[9] & 3) << 1) + 2;
                        cs = (csd[8] >> 6) + ((WORD)csd[7] << 2) + ((WORD)(csd[6] & 3) << 10) + 1;
                        *(LBA_t*)buff = cs << (n - 9);
                    }
                    res = RES_OK;
                }
            }
            DESELECT();
            break;

        case GET_BLOCK_SIZE:
            *(DWORD*)buff = 128;  /* 128 sectors (64KB) */
            res = RES_OK;
            break;

        default:
            res = RES_PARERR;
    }

    return res;
}

/* Get current time (stub: returns fixed date) */
DWORD get_fattime(void)
{
    /* Return FAT epoch: 1980-01-01 00:00:00 */
    return 0;
}
