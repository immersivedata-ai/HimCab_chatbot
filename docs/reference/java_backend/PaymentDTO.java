package com.edtech.anubudh.himcab.trip.dto;

import com.edtech.anubudh.himcab.core.model.NightCharges;
import com.edtech.anubudh.himcab.core.model.PaymentMode;
import com.edtech.anubudh.himcab.core.model.PaymentStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;

/**
 * @author Kartikey Gupta
 * @version 1.0.0
 * @since 06/09/2022
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class PaymentDTO {

    private Long id;

    private PaymentMode paymentMode;

    private PaymentStatus paymentStatus;

    private BigDecimal totalFare;

    private BigDecimal rideCharge;

    private BigDecimal bookingFee;

    private BigDecimal discount;

    private BigDecimal gst;

    private String paymentId;

    private Boolean status;

    private BigDecimal waitingCharges;

    private BigDecimal nightCharges;

    private BigDecimal additionKmFee;

    private BigDecimal baseKmAmount;

    private BigDecimal amountBeforeDiscount;
}