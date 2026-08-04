package com.edtech.anubudh.himcab.trip.controller;

import com.edtech.anubudh.himcab.core.dto.BaseDTO;
import com.edtech.anubudh.himcab.core.model.Notification;
import com.edtech.anubudh.himcab.core.model.PaymentMode;
import com.edtech.anubudh.himcab.core.model.TaxiType;
import com.edtech.anubudh.himcab.core.model.TripStatus;
import com.edtech.anubudh.himcab.core.model.user.TripType;
import com.edtech.anubudh.himcab.core.utils.APIVersion;
import com.edtech.anubudh.himcab.core.utils.ResponseEntityUtil;
import com.edtech.anubudh.himcab.trip.dto.NotificationDTO;
import com.edtech.anubudh.himcab.trip.dto.PaymentDTO;
import com.edtech.anubudh.himcab.trip.dto.TripDTO;
import com.edtech.anubudh.himcab.trip.service.NotificationService;
import com.edtech.anubudh.himcab.trip.service.TripService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.List;

/**
 * @author Kartikey Gupta
 * @version 1.0.0
 * @since 06/04/2022
 */

@RestController
@RequestMapping(APIVersion.RIDER_API_URL + "/trip")
@CrossOrigin
@Api("This end point is for get the trip related operation")
@RequiredArgsConstructor
public class TripUserController {

    private static final Logger log = LoggerFactory.getLogger(TripUserController.class);

    private final TripService tripService;

    private final ResponseEntityUtil responseEntityUtil;
    private final NotificationService notificationService;

    @GetMapping("/get-all")
    @ApiOperation(value = "This end point is to get the trips for rider", response = BaseDTO.class)
    public ResponseEntity<?> getAllTrips(@RequestParam("riderId") Long riderId, @RequestParam("page") int page, @RequestParam("size") int size, @RequestParam(value = "paymentMode", required = false)PaymentMode paymentMode, @RequestParam(value = "tripStatus" , required = false) TripStatus tripStatus, @RequestParam(value = "taxiType", required = false) TripType taxiType) {
        log.info("Requested Trip page is: {} and size is: {} and riderId is: {} and trip status is: {}", page, size, riderId, paymentMode);
        BaseDTO res = new BaseDTO();
        Pageable pageable = PageRequest.of(page, size);
        Page<TripDTO> data = tripService.getAllTrips(riderId, pageable, paymentMode, tripStatus, taxiType);
        res.getResponse(data);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @PostMapping("/book-ride")
    @ApiOperation(value = "This end point is to book the ride", response = BaseDTO.class)
    public ResponseEntity<?> bookARide(@RequestBody TripDTO tripDTO) {
        log.info("Requested Trip dto to book the ride is: {}", tripDTO);
        BaseDTO res = new BaseDTO();
        TripDTO data = tripService.bookARide(tripDTO);
        res.saveResponse(data);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @DeleteMapping("/cancel-ride/{id}")
    @ApiOperation(value = "This end point is to cancel the ride on rider side", response = BaseDTO.class)
    public ResponseEntity<?> cancelRide(@PathVariable("id") Long id, @RequestParam("reason") String reason) {
        log.info("Requested id to cancel a ride is: {} with reason {}", id, reason);
        BaseDTO res = new BaseDTO();
        tripService.cancelRide(id, reason);
        res.deleteResponse();
        return responseEntityUtil.constructResponseEntity(res);
    }

    @GetMapping("/{id}")
    @ApiOperation(value = "This end point is to get trip  for rider", response = BaseDTO.class)
    public ResponseEntity<?> getTrip(@PathVariable("id") Long id) {
        log.info("Requested id to get trip details is: {}", id);
        BaseDTO res = new BaseDTO();
        TripDTO tripDTO = tripService.getById(id);
        res.getResponse(tripDTO);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @GetMapping("/assign-driver/{tripId}")
    @ApiOperation(value = "This end point is to assign driver on rider side", response = BaseDTO.class)
    public ResponseEntity<?> assignDriver(@PathVariable("tripId") Long tripId) {
        log.info("Requested trip id to check assign driver is: {}", tripId);
        BaseDTO res = new BaseDTO();
        TripDTO data = tripService.assignDriver(tripId);
        res.getResponse(data);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @PatchMapping("/payment/{tripId}")
    @ApiOperation(value = "This end point is to update Payment on rider side", response = BaseDTO.class)
    public ResponseEntity<?> updatePayment(@PathVariable("tripId") Long tripId, @RequestBody PaymentDTO paymentDTO) {
        log.info("Requested trip id to update payment is: {}", tripId);
        BaseDTO res = new BaseDTO();
        TripDTO data = tripService.updatePayment(tripId, paymentDTO);
        res.getResponse(data);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @PutMapping("/update-rating/{tripId}")
    @ApiOperation(value = "This end point is to update rating on rider side", response = BaseDTO.class)
    public ResponseEntity<?> updateRating(@PathVariable("tripId") Long tripId, @RequestParam("rating") BigDecimal tripRating) {
        log.info("Requested trip id to update rating is: {}", tripId);
        BaseDTO res = new BaseDTO();
        TripDTO data = tripService.updateRating(tripId, tripRating);
        res.saveResponse(data);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @GetMapping("/get-all/{riderId}")
    @ApiOperation(value = "This end point is to get all notification for rider", response = BaseDTO.class)
    public ResponseEntity<?> getAllNotification(@PathVariable("riderId") Long riderId) {
        log.info("Requested user id for get all notification is: {}", riderId);
        BaseDTO res = new BaseDTO();
        List<NotificationDTO> allNotification = notificationService.getAllNotification(riderId);
        res.getResponse(allNotification);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @DeleteMapping("/notification/{id}")
    @ApiOperation(value = "This end point is used for delete Notification", response = BaseDTO.class)
    public ResponseEntity<?> deleteNotification(@PathVariable("id") Long id, @RequestParam("riderId") Long riderId) {
        BaseDTO res = new BaseDTO();
        log.info("Requested Notification id is: {} for rider: {}.", id, riderId);
        notificationService.deleteNotificationForRider(id, riderId);
        res.deleteResponse();
        return responseEntityUtil.constructResponseEntity(res);
    }

    @DeleteMapping("/all-notification/{riderId}")
    @ApiOperation(value = "This end point is used for delete all Notification by riderId", response = BaseDTO.class)
    public ResponseEntity<?> deleteAllNotification(@PathVariable("riderId") Long riderId) {
        BaseDTO res = new BaseDTO();
        log.info("Requested rider id is: {} to delete all notification", riderId);
        notificationService.deleteAllNotification(riderId);
        res.deleteResponse();
        return responseEntityUtil.constructResponseEntity(res);
    }

    @DeleteMapping("/delete-trip/{id}")
    @ApiOperation(value = "This end point is used for delete rider trip", response = ResponseEntity.class)
    public ResponseEntity<?> deleteTrip(@PathVariable("id") Long id) {
        BaseDTO res = new BaseDTO();
        log.info("Requested trip id is: {}", id);
        tripService.deleteTrip(id);
        res.deleteResponse();
        return responseEntityUtil.constructResponseEntity(res);
    }
    @PutMapping("/update-riderNotification-status/{notificationId}")
    @ApiOperation(value = "This end point is to update notification for rider", response = BaseDTO.class)
    public ResponseEntity<?> updateNotificationStatus(@PathVariable("notificationId") Long notificationId,@RequestParam("status") Boolean status) {
        log.info("Requested notification id is{} for change notification status is: {}", notificationId,status);
        BaseDTO res = new BaseDTO();
        NotificationDTO allNotification = notificationService.updateDriverNotificationStatus(notificationId,status);
        res.getResponse(allNotification);
        return responseEntityUtil.constructResponseEntity(res);
    }

    @GetMapping("/checking-trip-cancel/{id}")
    @ApiOperation(value = "This end point is to get trip  for rider", response = BaseDTO.class)
    public ResponseEntity<?> getTripForCheckingCancel(@PathVariable("id") Long id) {
        log.info("Requested id to get trip details is: {}", id);
        BaseDTO res = new BaseDTO();
        TripDTO tripDTO = tripService.getTripForCheckingCancel(id);
        res.getResponse(tripDTO);
        return responseEntityUtil.constructResponseEntity(res);
    }
}