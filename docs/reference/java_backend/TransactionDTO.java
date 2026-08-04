package com.edtech.anubudh.himcab.trip.dto;

import com.edtech.anubudh.himcab.core.model.PaymentMode;
import com.edtech.anubudh.himcab.core.model.PaymentStatus;
import lombok.Data;

import java.math.BigDecimal;
import java.util.Date;

@Data
public class TransactionDTO {

    private String paymentId;

    private String transactionId;

    private BigDecimal transactionAmount;

    private String dateOfTransaction;

    private String timeOfTransaction;

    private PaymentMode paymentMode;

    private PaymentStatus paymentStatus;

    private Long tripId;

    private String driverName;

    private String coupon;

    private BigDecimal gst;

    private String toAddress;

    private String fromAddress;

    private String vehicleNumber;

    private String invoice;

    private String tripType;

    private String stop1;

    private String stop2;
}