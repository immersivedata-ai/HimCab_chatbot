package com.edtech.anubudh.himcab.trip.dto;

import com.edtech.anubudh.himcab.core.model.TripStatus;
import com.edtech.anubudh.himcab.core.model.user.TripType;
import lombok.Builder;
import lombok.Data;

import java.math.BigDecimal;
import java.util.Date;
import java.util.List;

/**
 * @author Kartikey Gupta
 * @version 1.0.0
 * @since 06/04/2022
 */

@Data
public class TripDTO extends RentCarDTO {

    private DirectionDTO fromDirectionAddress;

    private List<DirectionDTO> toDirectionAddress;

    private BigDecimal tripRating;

    private String reachOnStartingLocationTime;

    @Builder(builderMethodName = "TripBuilder")
    public TripDTO(Long id, Long tripId, Long categoryId, String toAddress, String fromAddress, String latitude, String longitude, Date date, String bookingTime, Long rentHours, String categoryName, TripStatus tripStatus, Boolean status, BigDecimal distance, TripType tripType, Long riderId, Long driverId, Long otp,String otpTime, PaymentDTO paymentDTO, String userName, String userNumber, String userEmail, String userImage, String bookingType, BigDecimal startReading, BigDecimal endReading, String startReadingImage, String endReadingImage, String startTime, String endTime, String startDate, String endDate, DirectionDTO fromDirectionAddress, List<DirectionDTO> toDirectionAddress, Long offerId, BigDecimal tripRating, String reachOnStartingLocationTime) {
        super(id, tripId, categoryId,offerId, toAddress, fromAddress, latitude, longitude, date, bookingTime, rentHours, categoryName, tripStatus, status, distance, tripType, riderId, driverId, otp,otpTime, paymentDTO, userName, userNumber, userEmail, userImage, bookingType, startReading, endReading, startReadingImage, endReadingImage,startTime,endTime,startDate,endDate );
        this.fromDirectionAddress = fromDirectionAddress;
        this.toDirectionAddress = toDirectionAddress;
        this.tripRating = tripRating;
        this.reachOnStartingLocationTime = reachOnStartingLocationTime;
    }
}