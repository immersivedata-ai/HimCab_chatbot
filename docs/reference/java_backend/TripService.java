package com.edtech.anubudh.himcab.trip.service.impl;

import com.edtech.anubudh.himcab.core.dto.BaseDTO;
import com.edtech.anubudh.himcab.core.exception.AlreadyPresentException;
import com.edtech.anubudh.himcab.core.exception.HimCabIndiaException;
import com.edtech.anubudh.himcab.core.exception.ResourceNotFoundException;
import com.edtech.anubudh.himcab.core.model.*;
import com.edtech.anubudh.himcab.core.model.user.Driver;
import com.edtech.anubudh.himcab.core.model.user.Rider;
import com.edtech.anubudh.himcab.core.model.user.TripType;
import com.edtech.anubudh.himcab.core.repository.*;
import com.edtech.anubudh.himcab.core.service.GCPImageUploadService;
import com.edtech.anubudh.himcab.core.service.WalletUpdateService;
import com.edtech.anubudh.himcab.trip.dto.*;
import com.edtech.anubudh.himcab.trip.dto.mapresponseDTO.Root;
import com.edtech.anubudh.himcab.trip.service.NotificationService;
import com.edtech.anubudh.himcab.trip.service.PushNotificationService;
import com.edtech.anubudh.himcab.trip.service.TripService;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.storage.*;
import com.google.gson.Gson;
import lombok.RequiredArgsConstructor;
import net.minidev.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Service;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.xhtmlrenderer.pdf.ITextRenderer;

import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * @author Kartikey Gupta
 * @version 1.0.0
 * @since 06/04/2022
 */

@Service
@RequiredArgsConstructor
public class TripServiceImpl implements TripService {

    private static final Logger log = LoggerFactory.getLogger(TripServiceImpl.class);

    private final TripRepository tripRepository;

    private final CategoryRepository categoryRepository;

    private final RiderRepository riderRepository;

    private final OfferRepository offerRepository;

    private final PushNotificationService pushNotificationService;

    private final DriverRepository driverRepository;

    private final DriverBookingAuditRepository driverBookingAuditRepository;

    private final DriverLiveLocationRepository driverLiveLocationRepository;

    private final NotificationRepository notificationRepository;

    private final NotificationService notificationService;
    private final WalletRepository walletRepository;
    @Value("${anubudh.driver.range}")
    private String rangeDistance;
    @Value("${anubudh.googlemap.baseurl}")
    private String BaseURL;
    @Value("${gcp.bucket.name}")
    private String gcpBucketName;
    @Value("${gcp.project.id}")
    private String gcpProjectId;
    @Value("${anubudh.himcab.logo}")
    private String logo;
    private final GCPImageUploadService gcpImageUploadService;
    private final RentCarRepository rentCarRepository;
    @Value("${gcp.save.image.path}")
    private String gcpUrl;

    private final WalletUpdateService walletUpdateService;
    private final AdminRepository adminRepository;
    private final DriverTransactionRepository driverTransactionRepository;
    private final DriverWalletTransactionRepository driverWalletTransactionRepository;
    private final FareRepository fareRepository;
    private final PaymentRepository paymentRepository;

    public static Long generateOTP() {
        int randomPin = (int) (Math.random() * 9000) + 1000;
        return Long.valueOf(randomPin);
    }

    /**
     * @param riderId
     * @param pageable
     * @return
     */
    @Override
    public Page<TripDTO> getAllTrips(Long riderId, Pageable pageable, PaymentMode paymentMode, TripStatus tripStatus,
            TripType taxiType) {
        Specification<Trip> spec = Specification.where(null);
        if (riderId != null) {
            spec = spec.and((root, query, builder) -> builder.equal(root.get("rider").get("id"), riderId));
        }
        if (tripStatus != null) {
            switch (tripStatus) {
                case COMPLETED:
                    spec = spec.and((root, query, builder) -> builder.equal(root.get("tripStatus"), tripStatus));
                    break;
                case ONGOING:
                case REACH_ON_DESTINATION:
                case REACH_ON_START_LOCATION:
                    spec = spec.and((root, query, builder) -> builder.or(
                            builder.equal(root.get("tripStatus"), TripStatus.ONGOING),
                            builder.equal(root.get("tripStatus"), TripStatus.REACH_ON_DESTINATION),
                            builder.equal(root.get("tripStatus"), TripStatus.REACH_ON_START_LOCATION)));
                    break;
                case UPCOMING:
                case BOOKED:
                case NOT_BOOKED:
                    spec = spec.and((root, query, builder) -> builder.or(
                            builder.equal(root.get("tripStatus"), TripStatus.UPCOMING),
                            builder.equal(root.get("tripStatus"), TripStatus.BOOKED),
                            builder.equal(root.get("tripStatus"), TripStatus.NOT_BOOKED)));
                    break;
                case CANCELLED:
                case REJECTED:
                    spec = spec.and((root, query, builder) -> builder.or(
                            builder.equal(root.get("tripStatus"), TripStatus.CANCELLED),
                            builder.equal(root.get("tripStatus"), TripStatus.REJECTED)));
                    break;
            }
        }
        if (paymentMode != null) {
            spec = spec
                    .and((root, query, builder) -> builder.equal(root.get("payment").get("paymentMode"), paymentMode));
        }
        if (taxiType != null) {
            spec = spec.and((root, query, builder) -> builder.equal(root.get("tripType"), taxiType));
        }
        pageable = PageRequest.of(pageable.getPageNumber(), pageable.getPageSize(),
                Sort.by(Sort.Direction.DESC, "scheduleDate"));
        Page<Trip> trips = tripRepository.findAll(spec, pageable);
        return trips.map(this::convertTrip);
    }

    /**
     * @param tripDTO
     * @return
     */
    @Override
    public TripDTO bookARide(TripDTO tripDTO) {
        Trip save = tripRepository.save(convertTripDTO(tripDTO));
        log.info("Ride booked successfully with id: {}", save.getId());
        return convertTrip(save);
    }

    public void notifyAllActiveDrivers(Trip trip, String deviceId) {
        Rider rider = trip.getRider();
        Map<String, String> pushData = new HashMap<>();
        pushData.put("tripId", String.valueOf(trip.getId()));
        pushData.put("customerName", rider.getFirstName() + " " + rider.getLastName());
        pushData.put("datetime", trip.getScheduleDate().toString() + " " + trip.getScheduleTime().toString());
        pushData.put("time", trip.getScheduleTime().toString());
        pushData.put("date", trip.getScheduleDate().toString());
        pushData.put("tripType", trip.getTripType().toString());
        pushData.put("toAddress", trip.getToAddress().toString());
        pushData.put("fromAddress", trip.getFromAddress().getAddress());
        pushData.put("notificationType", NotificationType.TRIP_BOOKING.toString());
        pushData.put("title", "New Booking");
        pushData.put("body", "You got a new booking!");
        pushData.put("topic", "Hurrahhh! you have a new Booking");
        pushNotificationService.sendPushNotification(pushData, deviceId);
    }

    /**
     * @param id
     * @return
     */
    @Override
    public TripDTO getById(Long id) {
        return convertTrip(
                tripRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Trip", "id", id)));
    }

    @Override
    public TripDTO getTripById(Long id) {
        Trip trip = tripRepository.findTripById(id);
        boolean isDeleted = trip.getIsDeleted();
        if (isDeleted) {
            BaseDTO res = new BaseDTO();
            res.failedResponse("Trip Deleted");
            return null;
        } else {
            return convertTrip(trip);
        }
    }

    /**
     * @param id
     * @param reason
     */
    @Override
    public void cancelRide(Long id, String reason) {
        Trip trip = tripRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Booking", "id", id));
        trip.setTripStatus(TripStatus.CANCELLED);
        trip.setReason(reason);
        tripRepository.save(trip);
        if (trip.getDriver() != null) {
            Driver driver = trip.getDriver();
            String deviceId = "";
            if (driver.getDeviceId() != null) {
                deviceId = driver.getDeviceId();
            }
            notificationService.saveNotification(convertNotificationForDriver(trip, deviceId));
            notifyTripCancelledToDriver(trip, deviceId);
        }
    }

    private Notification convertNotificationForDriver(Trip trip, String deviceId) {
        Driver driver = trip.getDriver();
        Notification notification = new Notification();
        notification.setTrip(trip);
        notification.setDriver(driver);
        notification.setDeviceId(deviceId);
        notification.setName(driver.getFirstName() + " " + driver.getLastName());
        notification.setNotificationType(NotificationType.TRIP_CANCEL);
        notification.setTitle("Himcab");
        notification.setStatus(true);
        notification.setMessage(trip.getRider().getFirstName() + " cancel the ride");
        return notification;
    }

    private void notifyTripCancelledToDriver(Trip trip, String deviceId) {
        Rider rider = trip.getRider();
        Driver driver = trip.getDriver();
        Map<String, String> pushData = new HashMap<>();
        pushData.put("tripId", String.valueOf(trip.getId()));
        pushData.put("driverName", driver.getFirstName() + " " + driver.getLastName());
        pushData.put("notificationType", NotificationType.TRIP_CANCEL.toString());
        String riderName;
        if (rider.getFirstName() != null && !rider.getFirstName().isEmpty()) {
            riderName = rider.getFirstName();
        } else {
            riderName = rider.getPhoneNumber();
        }
        String message = "Trip Booking Cancelled by " + riderName;
        pushData.put("title", "Himcab");
        pushData.put("body", message);
        pushNotificationService.sendPushNotification(pushData, deviceId);
    }

    /**
     * @param id
     * @param reason
     */
    @Override
    public void cancelRideByDriver(Long id, String reason) {
        Trip trip = tripRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Booking", "id", id));
        trip.setTripStatus(TripStatus.REJECTED);
        trip.setReason(reason);
        tripRepository.save(trip);
    }

    /**
     * @param tripId
     * @return
     */
    @Override
    public TripDTO assignDriver(Long tripId) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        Boolean isAssigned = false;
        String deviceId = "";
        Pageable pageable = PageRequest.of(0, 30);
        Optional<List<Driver>> allDrivers = driverRepository.findAllByIsDeletedAndIsActiveAndOnlineStatus(false, true,
                true, pageable);
        if (allDrivers.isPresent()) {
            for (Driver driver : allDrivers.get()) {
                Long driverId = driver.getId();
                Boolean isDriverInRange = findDriverRange(driverId, tripId);
                if (isDriverInRange) {
                    if (isVehicleMakeMatching(trip, driver) && isDriverTripTypeMatching(trip, driver)) {
                        log.info("This is a driver with id: " + driver.getId() + " and device email: "
                                + driver.getEmail());
                        Optional<DriverBookingAudit> audit = driverBookingAuditRepository
                                .findByDriver_IdAndTrip_IdAndIsDeclined(driver.getId(), tripId, true);

                        if (!isAssigned && !audit.isPresent()) {
                            // Check if the driver has both local and outstation trips ongoing
                            if (hasDriverBothTripTypesOngoing(driver)) {
                                log.info(
                                        "Driver {} has ongoing trips of both local and outstation types. Skipping notification.",
                                        driver.getId());
                                continue; // Skip sending notification to this driver
                            }
                            if (driver.getDeviceId() != null && !driver.getDeviceId().isEmpty()) {
                                deviceId = driver.getDeviceId();
                            } else {
                                continue;
                            }
                            if (isVehicleMakeMatching(trip, driver)) {
                                Optional<Notification> notifiedDrivers = notificationRepository
                                        .findByDevice_idAndTrip_Id(tripId, deviceId);

                                if (!notifiedDrivers.isPresent()) {
                                    log.info("notification saved for trip id is {}", tripId);
                                    notificationService.saveNotification(convertNotification(trip, deviceId));
                                    notifyAllActiveDrivers(trip, deviceId);
                                }
                            }
                        }
                        isAssigned = checkAssignDriverForTrip(tripId);
                        if (isAssigned) {
                            return convertTrip(tripRepository.findById(tripId).get());
                        }
                    }
                }
            }
        }
        return null;
    }

    private boolean hasDriverBothTripTypesOngoing(Driver driver) {
        List<Trip> ongoingLocalTrips = tripRepository.findAllByDriverAndTripTypeAndTripStatus(driver, TripType.LOCAL,
                TripStatus.ONGOING);
        List<Trip> ongoingOutstationTrips = tripRepository.findAllByDriverAndTripTypeAndTripStatus(driver,
                TripType.OUTSTATION, TripStatus.ONGOING);
        return !ongoingLocalTrips.isEmpty() || !ongoingOutstationTrips.isEmpty();
    }

    private boolean isDriverTripTypeMatching(Trip trip, Driver driver) {
        if (driver.getVehicle() != null && driver.getVehicle().getTaxiServiceType() != null) {
            List<TaxiServiceType> taxiServiceTypes = driver.getVehicle().getTaxiServiceType();
            for (TaxiServiceType taxiServiceType : taxiServiceTypes) {
                if (taxiServiceType.getTaxiType() != null
                        && isMatchingTripType(trip.getTripType(), taxiServiceType.getTaxiType())) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean isMatchingTripType(TripType tripType, TaxiType taxiType) {
        switch (tripType) {
            case LOCAL:
                return taxiType == TaxiType.LOCAL_TAXI;
            case RENTAL:
                return taxiType == TaxiType.RENTAL_TAXI;
            case OUTSTATION:
                return taxiType == TaxiType.OUTSTATION_TAXI;
            default:
                return false;
        }
    }

    private boolean isVehicleMakeMatching(Trip trip, Driver driver) {
        List<String> tripVehicleModels = trip.getCategory().getVehicleModel();
        String driverVehicleModel = driver.getVehicle().getVehicleModel();
        boolean result = tripVehicleModels.contains(driverVehicleModel);
        log.info("Trip Vehicle Models: {}", tripVehicleModels);
        log.info("Driver Vehicle Model: {}", driverVehicleModel);
        log.info("isVehicleMakeMatching Result: {}", result);
        return result;
    }

    private Notification convertNotification(Trip trip, String deviceId) {
        Notification notification = new Notification();
        notification.setTrip(trip);
        notification.setDeviceId(deviceId);
        return notification;
    }

    private Boolean findDriverRange(Long driverId, Long tripId) {
        Trip trip = tripRepository.findById(tripId).get();
        Direction direction = trip.getFromAddress();
        String tripLat = direction.getLatitude();
        String tripLon = direction.getLongitude();
        Long driverRangeKm;
        switch (trip.getTripType()) {
            case LOCAL:
                driverRangeKm = adminRepository.getDriverRange();
                break;
            case RENTAL:
                driverRangeKm = adminRepository.getDriverRangeForRental();
                break;
            case OUTSTATION:
                driverRangeKm = adminRepository.getDriverRangeForOutstation();
                break;
            default:
                driverRangeKm = adminRepository.getDriverRange();
        }

        List<DriverLiveLocation> driverLocations = driverLiveLocationRepository.findByDriverId(driverId);
        if (driverLocations.isEmpty()) {
            return false;
        }
        DriverLiveLocation driverLiveLocation = driverLocations.get(0);
        String driverLat = driverLiveLocation.getLatitude();
        String driverLon = driverLiveLocation.getLongitude();

        String URL = BaseURL +
                "&origin=" +
                driverLat + "," + driverLon +
                "&destination=" +
                tripLat + "," + tripLon +
                "&sensor=false&mode=driving&outputFormat=json&key=AIzaSyBHJ339Q0UfErQfihMr1qEi7JY7V07bn8M";

        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(5000);
        RestTemplate restTemplate = new RestTemplate(factory);
        JSONObject res = restTemplate.getForObject(URL, JSONObject.class);
        Gson gson = new Gson();
        Root object = gson.fromJson(res.toString(), Root.class);
        if (!object.routes.isEmpty()) {
            String distance_text = object.routes.get(0).legs.get(0).distance.text;
            String distance_value = object.routes.get(0).legs.get(0).distance.value;
            long distanceMeters = Long.parseLong(distance_value);
            long driverRangeMeters = driverRangeKm * 1000;
            log.info("Distance value is {} and driver range is{}", distanceMeters, driverRangeMeters);
            return distanceMeters < driverRangeMeters;
        } else {
            return false;
        }
    }

    private Boolean checkAssignDriverForTrip(Long tripId) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        if (trip.getDriver() != null && trip.getTripStatus() == TripStatus.BOOKED) {
            log.info("Driver assign successfully with id: {} for trip id: {}", trip.getDriver().getId(), tripId);
            return true;
        }
        return false;
    }

    private TripDTO convertTrip(Trip save) {
        String image = "";
        if (save.getRider().getUserImage() != null && !Objects.equals(save.getRider().getUserImage(), "")) {
            image = gcpUrl + save.getRider().getUserImage();
        }
        RentCar rentCar = rentCarRepository.findByTripId(save.getId());
        Long renalHours = 0L;
        if (rentCar != null) {
            renalHours = rentCar.getRentHours();
        }
        BigDecimal endReading = null;
        String endReadingImage = null;
        String endTime = null;
        String endDate = null;
        String startDate = null;
        String startTime = null;
        BigDecimal startReading = null;
        String startReadingImage = null;
        if (save.getOdometer() != null) {
            if (save.getOdometer().getEndReadingImage() != null
                    && !Objects.equals(save.getOdometer().getEndReadingImage(), "")) {
                endReadingImage = gcpUrl + save.getOdometer().getEndReadingImage();
            }
            if (save.getOdometer().getStartReadingImage() != null
                    && !Objects.equals(save.getOdometer().getStartReadingImage(), "")) {
                startReadingImage = gcpUrl + save.getOdometer().getStartReadingImage();
            }
            endReading = save.getOdometer().getEndReading();
            startReading = save.getOdometer().getStartReading();
            if (save.getOdometer().getEndTime() != null) {
                endTime = save.getOdometer().getEndTime();
            }

            if (save.getOdometer().getEndDate() != null) {
                endDate = save.getOdometer().getEndDate();
            }

            if (save.getOdometer().getStartDate() != null) {
                startDate = save.getOdometer().getStartDate();
            }

            if (save.getOdometer().getStartTime() != null) {
                startTime = save.getOdometer().getStartTime();
            }
        }
        Long driverId = null;
        PaymentDTO paymentDTO = null;
        if (save.getDriver() != null) {
            driverId = save.getDriver().getId();
        }
        if (save.getPayment() != null) {
            paymentDTO = convertPaymentDTO(save.getPayment());
        }
        Long categoryId = null;
        String categoryName = null;
        if (save.getCategory() != null) {
            categoryId = save.getCategory().getId();
            categoryName = save.getCategory().getName();
        }
        return TripDTO.TripBuilder()
                .bookingTime(save.getScheduleTime().toString())
                .date(save.getScheduleDate())
                .distance(save.getDistance())
                .categoryId(categoryId)
                .categoryName(categoryName)
                .id(save.getId())
                .tripStatus(save.getTripStatus())
                .fromDirectionAddress(convertDirection(save.getFromAddress()))
                .toDirectionAddress(convertDirection(save.getToAddress()))
                .otp(save.getOtp())
                .driverId(driverId)
                .riderId(save.getRider().getId())
                .rentHours(renalHours)
                .paymentDTO(paymentDTO)
                .userName(save.getRider().getFirstName() + " " + save.getRider().getLastName())
                .userNumber(save.getRider().getPhoneNumber())
                .userEmail(save.getRider().getEmail())
                .userImage(image)
                .tripType(save.getTripType())
                .offerId(save.getOffer() != null ? save.getOffer().getId() : null)
                .tripRating((save.getTripRating()))
                .reachOnStartingLocationTime(save.getReachOnStartingLocationTime())
                .endReading(endReading)
                .endReadingImage(endReadingImage)
                .endTime(endTime)
                .endDate(endDate)
                .startDate(startDate)
                .startTime(startTime)
                .startReading(startReading)
                .startReadingImage(startReadingImage)
                .build();
    }

    private PaymentDTO convertPaymentDTO(Payment payment) {
        return PaymentDTO.builder()
                .id(payment.getId())
                .paymentMode(payment.getPaymentMode())
                .paymentId(payment.getPaymentId())
                .paymentStatus(payment.getPaymentStatus())
                .bookingFee(payment.getBookingFee())
                .gst(payment.getGst())
                .discount(payment.getDiscount())
                .rideCharge(payment.getRideCharge())
                .status(payment.getIsActive())
                .totalFare(payment.getTotalFare())
                .nightCharges(payment.getNightCharges())
                .waitingCharges(payment.getWaitingCharges())
                .baseKmAmount(payment.getBaseKmAmount())
                .additionKmFee(payment.getAdditionKmFee())
                .amountBeforeDiscount(payment.getTotalFare()
                        .add(payment.getDiscount() != null ? payment.getDiscount() : BigDecimal.ZERO))
                .build();

    }

    private List<DirectionDTO> convertDirection(List<Direction> toAddress) {
        List<DirectionDTO> res = new ArrayList<>();
        for (Direction d : toAddress) {
            res.add(convertDirection(d));
        }
        return res;
    }

    private DirectionDTO convertDirection(Direction fromAddress) {
        if (fromAddress != null) {
            return DirectionDTO.builder()
                    .address(fromAddress.getAddress())
                    .latitude(fromAddress.getLatitude())
                    .longitude(fromAddress.getLongitude())
                    .build();
        } else {
            return new DirectionDTO();
        }
    }

    private Trip convertTripDTO(TripDTO tripDTO) {
        Offer offer = null;
        if (tripDTO.getOfferId() != null) {
            offer = offerRepository.findById(tripDTO.getOfferId()).orElse(new Offer());
        }
        return Trip.builder()
                .isDeleted(false)
                .isActive(true)
                .tripStatus(tripDTO.getTripStatus())
                .category(categoryRepository.findById(tripDTO.getCategoryId()).get())
                .distance(tripDTO.getDistance())
                .fromAddress(convertDirectionDto(tripDTO.getFromDirectionAddress()))
                .tripType(tripDTO.getTripType())
                .rider(riderRepository.findById(tripDTO.getRiderId()).get())
                .scheduleDate(tripDTO.getDate())
                .scheduleTime(LocalTime.parse(tripDTO.getBookingTime()))
                .toAddress(convertTripDTO(tripDTO.getToDirectionAddress()))
                .offer(offer)
                .tripStatus(tripDTO.getTripStatus())
                .otp(generateOTP())
                .payment(convertPayment(tripDTO.getPaymentDTO(), tripDTO.getRiderId()))
                .build();
    }

    public Map<String, BigDecimal> calculateTotalFare(TripDTO tripDTO, Category category) {
        Map<String, BigDecimal> fareDetails = new HashMap<>();
        BigDecimal totalFare = BigDecimal.ZERO;
        BigDecimal offerValue = BigDecimal.ZERO;
        BigDecimal waitingCharges = BigDecimal.ZERO;
        BigDecimal gstAmount = BigDecimal.ZERO;
        BigDecimal nightCharges = BigDecimal.ZERO;
        BigDecimal bookingCharges = BigDecimal.ZERO;
        BigDecimal rideCharges = BigDecimal.ZERO;
        Offer offer = null;

        LocalTime bookingTime = LocalTime.parse(tripDTO.getBookingTime());
        DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        LocalDateTime dateTime = LocalDateTime.parse(tripDTO.getOtpTime(), dateTimeFormatter);
        LocalDateTime reachOnStartingLocationTime = LocalDateTime.parse(tripDTO.getReachOnStartingLocationTime(),
                dateTimeFormatter);
        LocalTime otpTime = dateTime.toLocalTime();
        LocalTime reachTime = reachOnStartingLocationTime.toLocalTime();

        if (tripDTO.getTripType().equals(TripType.LOCAL)) {
            Optional<Fare> fare = fareRepository.getCategoryByIdForLocal(category.getId());
            LocalFare localFare = fare.get().getLocalFares();
            BigDecimal distanceInKm = tripDTO.getDistance().divide(new BigDecimal("1000")); // Convert meters to
                                                                                            // kilometers

            if (distanceInKm.compareTo(new BigDecimal("6")) <= 0) {
                // Distance is 6 km or less
                totalFare = localFare.getFlatCharges().add(localFare.getBookingCharges());
                log.info("Distance is 6 km or less. Total Fare (Flat Charges + Booking Charges): {}", totalFare);
            } else {
                BigDecimal extraDistance = distanceInKm.subtract(new BigDecimal("6"));
                totalFare = localFare.getBaseFare().multiply(extraDistance).add(localFare.getFlatCharges())
                        .add(localFare.getBookingCharges());
                log.info(
                        "Distance is greater than 6 km. Total Fare (Base Fare * Distance + Flat Charges + Booking Charges): {}",
                        totalFare);
            }
            bookingCharges = localFare.getBookingCharges();
            waitingCharges = calculateWaitingCharges(reachTime, otpTime, localFare.getWaitingCharges());
            totalFare = totalFare.add(waitingCharges);

            if (isNightTime(bookingTime, localFare.getNightCharges())) {
                nightCharges = localFare.getNightCharges().getNightChargeAmount();
                totalFare = totalFare.add(nightCharges);
                log.info("Night charges added: {}", localFare.getNightCharges().getNightChargeAmount());
            }
            if (tripDTO.getOfferId() != null) {
                offer = offerRepository.findById(tripDTO.getOfferId()).orElse(new Offer());
                offerValue = offer.getValue();
            }
            totalFare = totalFare.subtract(offerValue);
            gstAmount = totalFare.multiply(localFare.getGst().divide(BigDecimal.valueOf(100)));
            rideCharges = totalFare;
            totalFare = totalFare.add(gstAmount);
        } else if (tripDTO.getTripType().equals(TripType.OUTSTATION)) {
            Optional<Fare> fare = fareRepository.getCategoryById(category.getId());
            OutstationFare outstationFare = fare.get().getOutstationFares();
            BigDecimal distanceInKm = tripDTO.getDistance().divide(new BigDecimal("1000"));
            totalFare = outstationFare.getBaseFare().multiply(distanceInKm).add(outstationFare.getBookingCharges());
            waitingCharges = calculateWaitingCharges(reachTime, otpTime, outstationFare.getWaitingCharges());
            totalFare = totalFare.add(waitingCharges);
            bookingCharges = outstationFare.getBookingCharges();
            if (isNightTime(bookingTime, outstationFare.getNightCharges())) {
                nightCharges = outstationFare.getNightCharges().getNightChargeAmount();
                totalFare = totalFare.add(nightCharges);
            }

            if (tripDTO.getOfferId() != null) {
                offer = offerRepository.findById(tripDTO.getOfferId()).orElse(new Offer());
                offerValue = offer.getValue();
            }
            totalFare = totalFare.subtract(offerValue);
            gstAmount = totalFare.multiply(outstationFare.getGst().divide(BigDecimal.valueOf(100)));
            rideCharges = totalFare;
            totalFare = totalFare.add(gstAmount);
        }
        fareDetails.put("totalFare", totalFare.setScale(2, RoundingMode.HALF_UP));
        fareDetails.put("waitingCharges", waitingCharges.setScale(2, RoundingMode.HALF_UP));
        fareDetails.put("gstAmount", gstAmount.setScale(2, RoundingMode.HALF_UP));
        fareDetails.put("nightCharges", nightCharges.setScale(2, RoundingMode.HALF_UP));
        fareDetails.put("offerValue", offerValue.setScale(2, RoundingMode.HALF_UP));
        fareDetails.put("bookingCharges", bookingCharges.setScale(2, RoundingMode.HALF_UP));
        fareDetails.put("rideCharges", rideCharges.setScale(2, RoundingMode.HALF_UP));
        return fareDetails;
    }

    private boolean isNightTime(LocalTime bookingTime, NightCharges nightCharges) {
        LocalTime nightStartTime = nightCharges.getNightChargeStartTime();
        LocalTime nightEndTime = nightCharges.getNightChargeEndTime();
        if (nightStartTime.isBefore(nightEndTime)) {
            return bookingTime.isAfter(nightStartTime) && bookingTime.isBefore(nightEndTime);
        } else {
            return bookingTime.isAfter(nightStartTime) || bookingTime.isBefore(nightEndTime);
        }
    }

    private BigDecimal calculateWaitingCharges(LocalTime reachOnStartingLocationTime, LocalTime otpTime,
            BigDecimal waitingChargeRate) {
        if (otpTime.isAfter(reachOnStartingLocationTime.plusMinutes(10))) {
            long waitingMinutes = Duration.between(reachOnStartingLocationTime.plusMinutes(10), otpTime).toMinutes();
            return waitingChargeRate.multiply(new BigDecimal(waitingMinutes));
        }
        return BigDecimal.ZERO;
    }

    private Payment convertPaymentCharges(PaymentDTO paymentDTO, Long riderId, Map<String, BigDecimal> fareDetails) {
        Long id = null;
        if (paymentDTO.getId() != null) {
            id = paymentDTO.getId();
        }

        if (paymentDTO.getPaymentMode().equals("WALLET")) {
            Wallet riderWallet = walletRepository.getWalletByRiderId(riderId);
            BigDecimal riderWalletAmount = riderWallet.getAmount();
            BigDecimal newAmount = riderWalletAmount.subtract(fareDetails.get("totalFare"));
            riderWallet.setAmount(newAmount);
            walletRepository.save(riderWallet);
        }

        return Payment.builder()
                .id(id)
                .paymentMode(paymentDTO.getPaymentMode())
                .bookingFee(fareDetails.get("bookingCharges"))
                .discount(fareDetails.get("offerValue"))
                .gst(fareDetails.get("gstAmount"))
                .rideCharge(fareDetails.get("rideCharges"))
                .totalFare(fareDetails.get("totalFare"))
                .paymentStatus(paymentDTO.getPaymentStatus())
                .paymentId(paymentDTO.getPaymentId())
                .nightCharges(fareDetails.get("nightCharges"))
                .waitingCharges(fareDetails.get("waitingCharges"))
                .isPaymentInit(false)
                .isActive(true)
                .isDeleted(false)
                .build();
    }

    private List<Direction> convertTripDTO(List<DirectionDTO> toAddress) {
        List<Direction> directions = new ArrayList<>();
        for (DirectionDTO dd : toAddress) {
            directions.add(convertDirectionDto(dd));
        }
        return directions;
    }

    /**
     * @param tripId
     * @param driverId
     * @param reason
     */
    @Override
    public void skipRideByDriver(Long tripId, Long driverId, String reason) {
        Calendar cal = Calendar.getInstance();
        Date currentDate = cal.getTime();
        cal.add(Calendar.MONTH, -1);
        Date previousDate = cal.getTime();
        log.info("Previous: {} and current date: {}", previousDate, currentDate);
        List<DriverBookingAudit> driverBookingAudits = driverBookingAuditRepository
                .findAllByDriver_IdAndCreatedAtBetween(driverId, previousDate, currentDate);
        // if (driverBookingAudits != null && !driverBookingAudits.isEmpty() &&
        // driverBookingAudits.size() >= 3) {
        // throw new HimCabIndiaException("driver cross the monthly limit to skip the
        // ride.");
        // } else {
        driverBookingAuditRepository.save(createDriverBookingAudit(tripId, driverId, reason));
        log.info("Trip id: {} skipped by driver id: {} successfully", tripId, driverId);
        // }
    }

    /**
     * @param tripId
     * @param driverId
     * @return
     */
    @Override
    public TripDTO startRideByDriver(Long tripId, Long driverId) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        trip.setTripStatus(TripStatus.ONGOING);
        Rider rider = trip.getRider();
        if (rider == null) {
            throw new ResourceNotFoundException("Rider", "id", trip.getRider().getId());
        }
        String otpTime = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        trip.setOtpTime(otpTime);
        TripDTO tripDTO = convertTrip(trip);
        tripDTO.setOtpTime(otpTime);
        Category category = categoryRepository.findById(trip.getCategory().getId())
                .orElseThrow(() -> new ResourceNotFoundException("Category", "id", trip.getCategory().getId()));
        Map<String, BigDecimal> fareDetails = calculateTotalFare(tripDTO, category);

        Payment payment = convertPaymentCharges(tripDTO.getPaymentDTO(), trip.getRider().getId(), fareDetails);

        Trip savedTrip = tripRepository.save(trip);
        paymentRepository.save(payment);
        log.debug("Saved trip with otpTime: {}", savedTrip.getOtpTime());
        if (trip.getTripStatus().equals(TripStatus.ONGOING)) {
            String deviceId = rider.getDeviceId() != null ? rider.getDeviceId() : "";
            notificationService.saveNotification(convertNotificationForRiderForStartRide(savedTrip, deviceId));
            notifyTripBookedToRiderForStartRide(savedTrip, deviceId);
        }
        return convertTrip(savedTrip);
    }

    private Notification convertNotificationForRiderForStartRide(Trip trip, String deviceId) {
        Notification notification = new Notification();
        notification.setTrip(trip);
        notification.setDeviceId(deviceId);
        Rider rider = trip.getRider();
        notification.setRider(rider);
        notification.setName(rider.getFirstName() + " " + rider.getLastName());
        notification.setNotificationType(NotificationType.TRIP_STARTED);
        notification.setTitle("Himcab");
        notification.setStatus(true);
        notification.setMessage(trip.getDriver().getFirstName() + " has started your ride");
        return notification;
    }

    private void notifyTripBookedToRiderForStartRide(Trip trip, String deviceId) {
        Rider rider = trip.getRider();
        Map<String, String> pushData = new HashMap<>();
        pushData.put("tripId", String.valueOf(trip.getId()));
        pushData.put("customerName", rider.getFirstName() + " " + rider.getLastName());
        pushData.put("notificationType", NotificationType.TRIP_STARTED.toString());
        String driverName = trip.getDriver().getFirstName();
        String message = driverName + " has started your ride";
        pushData.put("title", "Himcab");
        pushData.put("body", message);
        pushNotificationService.sendPushNotification(pushData, deviceId);
    }

    /**
     * @param tripId
     * @param paymentDTO
     * @return
     */
    @Override
    public TripDTO updatePayment(Long tripId, PaymentDTO paymentDTO) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        Long riderId = trip.getRider().getId();

        if (paymentDTO.getPaymentMode().equals(PaymentMode.WALLET)) {
            Wallet wallet = walletRepository.getWalletByRiderId(riderId);
            BigDecimal currAmt = wallet.getAmount();
            BigDecimal totalFare = paymentDTO.getTotalFare();

            if (totalFare.longValue() > currAmt.longValue()) {
                throw new HimCabIndiaException("Insufficient Funds in wallet!");
            }

            BigDecimal newAmt = currAmt.subtract(totalFare);
            wallet.setAmount(newAmt);
            walletRepository.save(wallet);
            trip.setTripStatus(TripStatus.COMPLETED);
            String deviceId = "";
            Driver driver = trip.getDriver();
            if (driver.getDeviceId() != null) {
                deviceId = driver.getDeviceId();
            }
            notificationService.saveNotification(convertNotificationForDriverForPayment(trip, deviceId));
            notifyTripBookedToDriverForPayment(trip, deviceId);
            log.info("trip notification is saved for driver id is {}", driver.getId());
        }
        trip.setPayment(convertPayment(paymentDTO, riderId));
        if (trip.getPayment().getPaymentMode() == PaymentMode.WALLET && trip.getPayment().getPaymentId() == null) {
            String dateStr = LocalDateTime.now(ZoneId.of("Asia/Kolkata"))
                    .format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
            trip.getPayment().setPaymentId("HC" + riderId + dateStr + tripId);
        }
        if (trip.getTripStatus() == TripStatus.COMPLETED && trip.getPayment().getPaymentMode() == PaymentMode.CASH) {
            trip.setTripStatus(TripStatus.COMPLETED);
            String deviceId = "";
            Driver driver = trip.getDriver();
            if (driver.getDeviceId() != null) {
                deviceId = driver.getDeviceId();
            }
            notificationService.saveNotification(convertNotificationForDriverForPayment(trip, deviceId));
            notifyTripBookedToDriverForPayment(trip, deviceId);
            log.info("trip notification is saved for driver id is {}", driver.getId());
        }
        trip.setTripStatus(TripStatus.COMPLETED);
        Trip save = tripRepository.save(trip);
        log.info("Trip payment successfully updated with riderId: {}", riderId);
        return convertTrip(save);
    }

    private Notification convertNotificationForDriverForPayment(Trip trip, String deviceId) {
        Driver driver = trip.getDriver();
        Notification notification = new Notification();
        notification.setTrip(trip);
        notification.setDeviceId(deviceId);
        notification.setDriver(driver);
        notification.setName(driver.getFirstName() + " " + driver.getLastName());
        notification.setNotificationType(NotificationType.PAYMENT_SUCCESSFUL);
        notification.setTitle("Himcab");
        notification.setStatus(true);
        notification.setMessage(trip.getRider().getFirstName() + " paid payment successfully");
        return notification;
    }

    private void notifyTripBookedToDriverForPayment(Trip trip, String deviceId) {
        Driver driver = trip.getDriver();
        Rider rider = trip.getRider();
        Map<String, String> pushData = new HashMap<>();
        pushData.put("tripId", String.valueOf(trip.getId()));
        pushData.put("driverName", driver.getFirstName() + " " + driver.getLastName());
        pushData.put("notificationType", NotificationType.PAYMENT_SUCCESSFUL.toString());
        String message = rider.getFirstName() + " paid payment successfully";
        pushData.put("title", "Himcab");
        pushData.put("body", message);
        pushNotificationService.sendPushNotification(pushData, deviceId);
    }

    /**
     * @param pageable
     * @return
     */
    @Override
    public Page<TripDTO> getAllTrips(Pageable pageable) {
        return tripRepository.findAllTripByIsDeletedOrderByIdDesc(false, pageable).map(this::convertTrips);
    }

    private TripDTO convertTrips(Trip save) {
        String image = "";
        if (save.getRider().getUserImage() != null && !Objects.equals(save.getRider().getUserImage(), "")) {
            image = gcpUrl + save.getRider().getUserImage();
        }
        Long driverId = null;
        if (save.getDriver() != null) {
            driverId = save.getDriver().getId();
        }
        return TripDTO.TripBuilder()
                .bookingTime(save.getScheduleTime().toString())
                .date(save.getScheduleDate())
                .id(save.getId())
                .tripStatus(save.getTripStatus())
                .otp(save.getOtp())
                .driverId(driverId)
                .riderId(save.getRider().getId())
                .userName(save.getRider().getFirstName() + " " + save.getRider().getLastName())
                .userNumber(save.getRider().getPhoneNumber())
                .userEmail(save.getRider().getEmail())
                .userImage(image)
                .tripType(save.getTripType())
                .tripRating((save.getTripRating()))
                .reachOnStartingLocationTime(save.getReachOnStartingLocationTime())
                .build();
    }

    @Override
    public List<DriverTransactionDTO> getDriverTransactions(Long id) {
        List<DriverTransaction> driverTransactionList = driverTransactionRepository.findByDriverId(id);
        return convertDriverTransactionList(driverTransactionList);
    }

    @Override
    public TripDTO getTripForCheckingCancel(Long id) {
        Trip trip = tripRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Booking", "id", id));
        trip.setTripStatus(TripStatus.SEARCHING_CANCELLED);
        Trip save = tripRepository.save(trip);
        return convertTrip(save);
    }

    private List<DriverTransactionDTO> convertDriverTransactionList(List<DriverTransaction> driverTransactionList) {
        List<DriverTransactionDTO> lessonDTOList = new ArrayList<>();
        for (DriverTransaction driverTransaction : driverTransactionList) {
            lessonDTOList.add(convertDriverTransactionDTO(driverTransaction));
        }
        return lessonDTOList;
    }

    private DriverTransactionDTO convertDriverTransactionDTO(DriverTransaction driverTransaction) {
        return DriverTransactionDTO.builder()
                .id(driverTransaction.getId())
                .driverId(driverTransaction.getTrip().getDriver().getId())
                .tripId(driverTransaction.getTrip().getId())
                .transactionId(driverTransaction.getTrip().getPayment().getTransactionId())
                .paymentId(driverTransaction.getTrip().getPayment().getPaymentId())
                .dateOfTransaction(driverTransaction.getDate())
                .transactionAmount(driverTransaction.getAmount())
                .driverAccountNumber(driverTransaction.getTrip().getDriver().getBankDetails().getAccountNumber())
                .paymentStatus(driverTransaction.getPaymentStatus().toString())
                .transactionUrl(generateInvoice(driverTransaction.getTrip().getDriver(), driverTransaction.getAmount(),
                        driverTransaction.getTrip().getPayment().getPaymentId(), driverTransaction.getDate()))
                .build();
    }

    public String generateInvoice(Driver driver, BigDecimal amount, String paymentId, String currentDate) {
        String randomKey = UUID.randomUUID().toString();
        String pdfKey = "transaction-invoice/" + randomKey + ".pdf";
        try {
            Storage storage = StorageOptions.newBuilder()
                    .setCredentials(GoogleCredentials.fromStream(new ClassPathResource("/key.json").getInputStream()))
                    .setProjectId(gcpProjectId)
                    .build()
                    .getService();

            BlobId blobId = BlobId.of(gcpBucketName, pdfKey);
            Blob blob = storage.get(blobId);
            if (blob != null && blob.exists()) {
                return gcpUrl + pdfKey;
            }
            ITextRenderer renderer = new ITextRenderer();
            String htmlContent = "<!DOCTYPE html>\n" +
                    "<html lang=\"en\">\n" +
                    "<head>\n" +
                    "    <title>Him Cab Invoice</title>\n" +
                    "    <style>\n" +
                    "        body {\n" +
                    "            font-family: Arial, sans-serif;\n" +
                    "        }\n" +
                    "        .invoice {\n" +
                    "            width: 80%;\n" +
                    "            margin: 0 auto;\n" +
                    "            border: 1px solid #ccc;\n" +
                    "            padding: 20px;\n" +
                    "        }\n" +
                    "        .header {\n" +
                    "            text-align: center;\n" +
                    "            font-size: 24px;\n" +
                    "            margin-bottom: 10px;\n" +
                    "        }\n" +
                    "        .logo {\n" +
                    "            text-align: center;\n" +
                    "        }\n" +
                    "        .logo img {\n" +
                    "            max-width: 100px; /* Adjust the image size as needed */\n" +
                    "        }\n" +
                    "        .center-text {\n" +
                    "            text-align: center;\n" +
                    "        }\n" +
                    "        .border-line {\n" +
                    "            border-top: 1px solid #ccc;\n" +
                    "            margin: 10px 0;\n" +
                    "        }\n" +
                    "        .details {\n" +
                    "            font-size: 16px;\n" +
                    "            margin-bottom: 10px;\n" +
                    "        }\n" +
                    "        /* Align labels to the left and values to the right */\n" +
                    "        .label {\n" +
                    "            display: inline-block;\n" +
                    "            width: 150px; /* Adjust as needed */\n" +
                    "            text-align: left;\n" +
                    "            font-weight: bold;\n" +
                    "        }\n" +
                    "        .value {\n" +
                    "            display: inline-block;\n" +
                    "            width: calc(80% - 80px); /* Adjust as needed */\n" +
                    "            text-align: right;\n" +
                    "        }\n" +
                    "    </style>\n" +
                    "</head>\n" +
                    "<body>\n" +
                    "    <div class=\"invoice\">\n" +
                    "        <div class=\"logo\">\n" +
                    "            <img src=\"" + logo + "\" alt=\"Him Cab Logo\"></img>\n" +
                    "        </div><br></br>\n" +
                    "        <div class=\"center-text\">\n" +
                    "            Receipt for the Driver, " + driver.getFirstName() + " " + driver.getLastName() + "\n" +
                    "        </div>\n" +
                    "        <div class=\"details\">\n" +
                    "            <div class=\"border-line\"></div>\n" +
                    "            <div class=\"label\">Date:</div><div class=\"value\">" + currentDate
                    + "</div><br></br><br></br>\n" +
                    "            <div class=\"label\">Payment ID:</div><div class=\"value\">" + paymentId + "</div>\n" +
                    "            <div class=\"border-line\"></div>\n" +
                    "            <div class=\"label\">Amount:</div><div class=\"value\">" + amount
                    + "</div><br></br><br></br>\n" +
                    "        </div>\n" +
                    "    </div>\n" +
                    "</body>\n" +
                    "</html>";

            renderer.setDocumentFromString(htmlContent);
            renderer.layout();

            ByteArrayOutputStream pdfOutputStream = new ByteArrayOutputStream();
            renderer.createPDF(pdfOutputStream);
            // Upload the PDF to GCP
            BlobInfo blobInfo = BlobInfo.newBuilder(blobId).setContentType("application/pdf").build();
            byte[] pdfBytes = pdfOutputStream.toByteArray();
            Blob createdBlob = storage.create(blobInfo, pdfBytes);
            log.info("File {} uploaded successfully.", pdfKey);
            log.debug("GCS bucket path is: {}", createdBlob.getMediaLink());
            return gcpUrl + pdfKey;
        } catch (Exception e) {
            e.printStackTrace();
            return "null";
        }
    }

    /**
     * @param tripId takes Long tripId.
     * @return get trip details by tripId.
     */
    @Override
    public TripDTO getReachOnDestination(Long tripId) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        trip.setTripStatus(TripStatus.REACH_ON_DESTINATION);
        Trip save = tripRepository.save(trip);
        return convertTrip(save);
    }

    @Override
    public TripDTO updatePaymentByDriver(Long tripId, TripStatus tripStatus, PaymentStatus paymentStatus) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        BigDecimal driverPercentage = adminRepository.getDriverPercentage();
        BigDecimal adminPercentage = BigDecimal.valueOf(100).subtract(driverPercentage);
        if (tripStatus == TripStatus.COMPLETED && trip.getPayment().getPaymentMode() == PaymentMode.CASH) {
            BigDecimal totalFare = trip.getPayment().getTotalFare();
            BigDecimal adminCommission = totalFare.multiply(
                    adminPercentage.divide(BigDecimal.valueOf(100), RoundingMode.HALF_UP));

            Wallet driverWallet = walletRepository.getWalletByDriverId(trip.getDriver().getId());
            if (driverWallet == null) {
                driverWallet = Wallet.builder()
                        .isActive(true)
                        .isDeleted(false)
                        .driver(trip.getDriver())
                        .amount(adminCommission)
                        .build();
                // Initialize walletUpdatedTrips map for new wallet
                if (driverWallet.getWalletUpdatedTrips() == null) {
                    driverWallet.setWalletUpdatedTrips(new HashMap<>());
                }
                driverWallet.getWalletUpdatedTrips().put(tripId, true);
                walletRepository.save(driverWallet);
            } else {
                // Initialize walletUpdatedTrips map if null
                if (driverWallet.getWalletUpdatedTrips() == null) {
                    driverWallet.setWalletUpdatedTrips(new HashMap<>());
                }
                // Guard: skip wallet update if this trip was already processed
                if (!driverWallet.getWalletUpdatedTrips().containsKey(tripId)) {
                    driverWallet.setAmount(driverWallet.getAmount().add(adminCommission));
                    driverWallet.getWalletUpdatedTrips().put(tripId, true);
                    walletRepository.save(driverWallet);
                }
            }

            // NEW: Save per-trip transaction with locked-in commission amount
            if (!driverWalletTransactionRepository.existsByTripId(tripId)) {
                DriverWalletTransaction walletTx = DriverWalletTransaction.builder()
                        .driver(trip.getDriver())
                        .trip(trip)
                        .amount(adminCommission)
                        .type(WalletTransactionType.COMMISSION)
                        .date(com.edtech.anubudh.himcab.core.utils.DateUtil.getCurrentDateString())
                        .paymentStatus(PaymentStatus.PENDING)
                        .isActive(true)
                        .isDeleted(false)
                        .build();
                driverWalletTransactionRepository.save(walletTx);
            }
        }
        trip.setTripStatus(tripStatus);
        trip.getPayment().setPaymentStatus(paymentStatus);
        if (tripStatus == TripStatus.COMPLETED && trip.getPayment().getPaymentMode() == PaymentMode.CASH) {
            String dateStr = LocalDateTime.now(ZoneId.of("Asia/Kolkata"))
                    .format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
            trip.getPayment().setPaymentId("HC" + trip.getRider().getId() + dateStr + tripId);
        }
        log.info("trip status is{} and payment status is {}", tripStatus, paymentStatus);
        Trip save = tripRepository.save(trip);
        if (trip.getTripStatus().equals(TripStatus.COMPLETED)) {
            String deviceId = "";
            Driver driver = trip.getDriver();
            if (driver.getDeviceId() != null) {
                deviceId = driver.getDeviceId();
            }
            notificationService.saveNotification(convertNotificationForDriverForPayment(trip, deviceId));
            notifyTripBookedToDriverForPayment(trip, deviceId);
            log.info("trip notification is saved for driver id is {}", driver.getId());
        }
        return convertTrip(save);
    }

    @Override
    public TripDTO updateRating(Long tripId, BigDecimal tripRating) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        trip.setTripRating(tripRating);
        Trip save = tripRepository.save(trip);
        return convertTrip(save);
    }

    @Override
    public TripDTO getReachOnStartingLocation(Long tripId, String reachOnStartLocationTime) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        trip.setTripStatus(TripStatus.REACH_ON_START_LOCATION);
        trip.setReachOnStartingLocationTime((reachOnStartLocationTime));
        Rider rider = trip.getRider();
        Trip save = tripRepository.save(trip);
        if (trip.getTripStatus().equals(TripStatus.REACH_ON_START_LOCATION)) {
            String deviceId = "";
            if (rider.getDeviceId() != null) {
                deviceId = rider.getDeviceId();
            }
            notificationService.saveNotification(convertNotificationForRiderForReachOn(trip, deviceId));
            notifyTripBookedToRiderForReachOn(trip, deviceId);
        }
        return convertTrip(save);
    }

    private Notification convertNotificationForRiderForReachOn(Trip trip, String deviceId) {
        Notification notification = new Notification();
        notification.setTrip(trip);
        notification.setDeviceId(deviceId);
        Rider rider = trip.getRider();
        notification.setRider(rider);
        notification.setName(rider.getFirstName() + " " + rider.getLastName());
        notification.setNotificationType(NotificationType.REACH_ON_START_LOCATION);
        notification.setTitle("Himcab");
        notification.setStatus(true);
        notification
                .setMessage(trip.getDriver().getFirstName() + " has reached on " + trip.getFromAddress().getAddress());
        return notification;
    }

    private void notifyTripBookedToRiderForReachOn(Trip trip, String deviceId) {
        Rider rider = trip.getRider();
        Driver driver = trip.getDriver();
        Map<String, String> pushData = new HashMap<>();
        pushData.put("tripId", String.valueOf(trip.getId()));
        pushData.put("customerName", rider.getFirstName() + " " + rider.getLastName());
        pushData.put("notificationType", NotificationType.REACH_ON_START_LOCATION.toString());
        String driverName = driver.getFirstName();
        String message = driverName + " has reached on " + trip.getFromAddress().getAddress();
        pushData.put("title", "Himcab");
        pushData.put("body", message);
        pushNotificationService.sendPushNotification(pushData, deviceId);
    }

    @Override
    public TripDTO updateWaitingCharges(Long tripId, BigDecimal waitingCharges) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        trip.getPayment().setWaitingCharges(waitingCharges);
        trip.getPayment().setTotalFare(trip.getPayment().getTotalFare().add(waitingCharges));
        Trip save = tripRepository.save(trip);
        return convertTrip(save);
    }

    @Override
    public void deleteTrip(Long id) {
        Trip trip = tripRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Trip", "id", id));
        trip.setIsDeleted(true);
        tripRepository.save(trip);
        List<Notification> notifications = notificationRepository.findByTripId(id);
        for (Notification notification : notifications) {
            notification.setIsDeleted(true);
            notificationRepository.save(notification);
        }
        log.info("Notification delete successfully with id: {}", id);
        log.info("Trip delete successfully with id: {}", id);
    }

    @Override
    public TripDTO updateStartOdometer(Long id, Long startReading, MultipartFile startReadingImage, String startTime,
            String startDate) {
        Trip trip = tripRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Trip", "id", id));
        if (trip.getOdometer() == null) {
            Odometer odometer = new Odometer();
            trip.setOdometer(odometer);
        }
        trip.getOdometer().setStartReading(BigDecimal.valueOf(startReading));
        trip.getOdometer().setStartReadingImage(gcpImageUploadService.uploadImageOnGCP(startReadingImage));
        trip.getOdometer().setStartTime(startTime);
        trip.getOdometer().setStartDate(startDate);
        trip.getOdometer().setIsActive(true);
        trip.getOdometer().setIsDeleted(false);
        Trip save = tripRepository.save(trip);
        log.info("Trip updated successfully with start odometer data");
        return convertTrip(save);
    }

    @Override
    public TripDTO updateEndOdometer(Long id, Long endReading, MultipartFile endReadingImage, String endTime,
            String endDate) {
        Trip trip = tripRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Trip", "id", id));
        if (trip.getOdometer() == null) {
            Odometer odometer = new Odometer();
            trip.setOdometer(odometer);
        }
        trip.getOdometer().setEndReading(BigDecimal.valueOf(endReading));
        trip.getOdometer().setEndReadingImage(gcpImageUploadService.uploadImageOnGCP(endReadingImage));
        trip.getOdometer().setEndTime(endTime);
        trip.getOdometer().setEndDate(endDate);
        Trip savedTrip = tripRepository.save(trip);
        log.info("Trip updated successfully with end odometer data");
        BigDecimal startReading = trip.getOdometer().getStartReading();
        if (startReading != null && startReading.compareTo(BigDecimal.ZERO) > 0) {
            Category category = categoryRepository.findById(trip.getCategory().getId())
                    .orElseThrow(() -> new ResourceNotFoundException("Category", "id", trip.getCategory().getId()));
            Map<String, BigDecimal> fareDetailsForRental = calculateTotalFareForRental(trip, startReading,
                    BigDecimal.valueOf(endReading), category);
            PaymentDTO paymentDTO = convertPaymentDTO(trip.getPayment());
            Payment payment = convertPaymentForRental(paymentDTO, trip.getRider().getId(), fareDetailsForRental);
            paymentRepository.save(payment);
            log.info("Payment saved successfully with calculated fare.");
        }
        return convertTrip(savedTrip);
    }

    public Map<String, BigDecimal> calculateTotalFareForRental(Trip trip, BigDecimal startReading,
            BigDecimal endReading, Category category) {
        Map<String, BigDecimal> fareDetails = new HashMap<>();
        BigDecimal totalFare = BigDecimal.ZERO;
        BigDecimal offerValue = BigDecimal.ZERO;
        BigDecimal nightCharges = BigDecimal.ZERO;
        BigDecimal gstAmount = BigDecimal.ZERO;
        BigDecimal rideCharges = BigDecimal.ZERO;
        BigDecimal baseKmAmount = BigDecimal.ZERO;
        BigDecimal additionKmFee = BigDecimal.ZERO;
        Offer offer = null;
        RentCar rentCar = rentCarRepository.findByTripId(trip.getId());
        LocalTime bookingTime = LocalTime.parse(rentCar.getBookingTime().toString());
        Long rentHours = rentCar.getRentHours();
        Optional<Fare> fareOptional = fareRepository.getCategoryByIdForRental(category.getId());

        if (fareOptional.isPresent()) {
            Fare fare = fareOptional.get();
            List<RentalChart> rentalCharts = fare.getRentalChartList();
            RentalChart matchingChart = findMatchingRentalChart(rentalCharts, rentHours);
            if (matchingChart != null) {
                if ((startReading != null && startReading.compareTo(BigDecimal.ZERO) != 0) &&
                        (endReading != null && endReading.compareTo(BigDecimal.ZERO) != 0)) {
                    BigDecimal baseFare = BigDecimal.valueOf(matchingChart.getFarePerKilometer())
                            .multiply(BigDecimal.valueOf(matchingChart.getMaxKilometer()));
                    log.info("Total fare after base fare calculation: {}", totalFare);
                    baseKmAmount = baseFare;
                    BigDecimal additionalFare = BigDecimal.ZERO;
                    BigDecimal distance = endReading.subtract(startReading);

                    if (distance.compareTo(BigDecimal.valueOf(matchingChart.getMaxKilometer())) > 0) {
                        BigDecimal additionalDistance = distance
                                .subtract(BigDecimal.valueOf(matchingChart.getMaxKilometer()));
                        additionalFare = BigDecimal.valueOf(matchingChart.getFareGreaterKilometer())
                                .multiply(additionalDistance);
                        log.info("Total fare after additional distance fare: {}", additionalFare);
                    }
                    totalFare = baseFare.add(additionalFare).setScale(2, RoundingMode.HALF_UP);
                    additionKmFee = additionalFare;
                    if (isNightTime(bookingTime, matchingChart.getNightCharges())) {
                        nightCharges = matchingChart.getNightCharges().getNightChargeAmount().setScale(2,
                                RoundingMode.HALF_UP);
                        totalFare = totalFare.add(nightCharges);
                        log.info("Night charges added: {}", nightCharges);
                    } else {
                        log.info("No night charges applied.");
                    }

                    if (rentCar.getOfferId() != null) {
                        offer = offerRepository.findById(rentCar.getOfferId()).orElse(new Offer());
                        offerValue = offer.getValue().setScale(2, RoundingMode.HALF_UP);
                        totalFare = totalFare.subtract(offerValue);
                    }
                    rideCharges = totalFare.setScale(2, RoundingMode.HALF_UP);
                    log.info("rideCharges : {}", rideCharges);
                    gstAmount = totalFare.multiply(matchingChart.getGst().divide(BigDecimal.valueOf(100)))
                            .setScale(2, RoundingMode.HALF_UP);
                    totalFare = totalFare.add(gstAmount).setScale(2, RoundingMode.HALF_UP);
                    log.info("Total fare after GST: {}", totalFare);
                }
            } else {
                log.error("No matching rental chart found for the given hours.");
            }
        } else {
            log.error("No fare found for the given category.");
        }
        fareDetails.put("totalFare", totalFare);
        fareDetails.put("nightCharges", nightCharges);
        fareDetails.put("gstAmount", gstAmount);
        fareDetails.put("rideCharges", rideCharges);
        fareDetails.put("offerValue", offerValue);
        fareDetails.put("baseKmAmount", baseKmAmount);
        fareDetails.put("additionKmFee", additionKmFee);
        return fareDetails;
    }

    private RentalChart findMatchingRentalChart(List<RentalChart> rentalCharts, Long rentHours) {
        for (RentalChart chart : rentalCharts) {
            if (Objects.equals(rentHours, chart.getMaxHour())) {
                log.info("renthr : {} and charthr: {}", rentHours, chart.getMaxHour());
                return chart;
            }
        }
        return null;
    }

    private Payment convertPaymentForRental(PaymentDTO paymentDTO, Long riderId, Map<String, BigDecimal> fareDetails) {
        Long id = null;
        if (paymentDTO.getId() != null) {
            id = paymentDTO.getId();
        }

        if (paymentDTO.getPaymentMode().equals(PaymentMode.WALLET)) {

            Wallet riderWallet = walletRepository.getWalletByRiderId(riderId);
            BigDecimal riderWalletAmount = riderWallet.getAmount();

            BigDecimal newAmount = riderWalletAmount.subtract(fareDetails.get("totalFare"));
            riderWallet.setAmount(newAmount);

            walletRepository.save(riderWallet);
        }
        return Payment.builder()
                .id(id)
                .paymentMode(paymentDTO.getPaymentMode())
                .bookingFee(paymentDTO.getBookingFee())
                .discount(fareDetails.get("offerValue"))
                .gst(fareDetails.get("gstAmount"))
                .rideCharge(fareDetails.get("rideCharges"))
                .totalFare(fareDetails.get("totalFare"))
                .paymentStatus(paymentDTO.getPaymentStatus())
                .paymentId(paymentDTO.getPaymentId())
                .nightCharges(fareDetails.get("nightCharges"))
                .waitingCharges(paymentDTO.getWaitingCharges())
                .baseKmAmount(fareDetails.get("baseKmAmount"))
                .additionKmFee(fareDetails.get("additionKmFee"))
                .isPaymentInit(false)
                .isActive(true)
                .isDeleted(false)
                .build();
    }

    @Override
    public Map<String, Integer> getTripsByMonth(String year) {
        Map<String, Integer> ridersCountByMonth = new LinkedHashMap<>();
        for (Long month = 1L; month <= 12; month++) {
            String monthString = String.format("%02d", month);
            LocalDate startOfMonth = LocalDate.parse(year + "-" + monthString + "-01");
            LocalDate endOfMonth = startOfMonth.withDayOfMonth(startOfMonth.lengthOfMonth());
            List<Trip> trips = tripRepository.findTripsByDateBetween(startOfMonth, endOfMonth);
            Long tripsCount = (long) trips.size();
            ridersCountByMonth.put(startOfMonth.getMonth().toString(), Math.toIntExact(tripsCount));
        }
        return ridersCountByMonth;
    }

    @Override
    public TripDTO updatePaymentByDriver(Long tripId, PaymentDTO paymentDTO) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        if (trip.getPayment() == null) {
            Payment payment = new Payment();
            trip.setPayment(payment);
        }
        trip.getPayment().setTotalFare(paymentDTO.getTotalFare());
        trip.getPayment().setDiscount(paymentDTO.getDiscount());
        trip.getPayment().setBookingFee(paymentDTO.getBookingFee());
        trip.getPayment().setGst(paymentDTO.getGst());
        trip.getPayment().setWaitingCharges(paymentDTO.getWaitingCharges());
        trip.getPayment().setNightCharges(paymentDTO.getNightCharges());
        trip.getPayment().setPaymentMode(paymentDTO.getPaymentMode());
        trip.getPayment().setPaymentStatus(paymentDTO.getPaymentStatus());
        trip.getPayment().setBaseKmAmount(paymentDTO.getBaseKmAmount());
        trip.getPayment().setAdditionKmFee(paymentDTO.getAdditionKmFee());
        trip.getPayment().setPaymentId(paymentDTO.getPaymentId());
        Trip save = tripRepository.save(trip);
        return convertTrip(save);
    }

    private Payment convertPayment(PaymentDTO paymentDTO, Long riderId) {
        Long id = null;
        BigDecimal rideCharge = null;
        BigDecimal bookingFee = null;
        BigDecimal additionKmFee = null;
        BigDecimal baseKmAmount = null;
        if (paymentDTO.getId() != null) {
            id = paymentDTO.getId();
            Payment existingPayment = paymentRepository.findById(id).orElse(null);
            if (existingPayment != null) {
                rideCharge = existingPayment.getRideCharge();
                bookingFee = existingPayment.getBookingFee();
                additionKmFee = existingPayment.getAdditionKmFee();
                baseKmAmount = existingPayment.getBaseKmAmount();
            }
        }

        if (paymentDTO.getPaymentMode().equals("WALLET")) {

            Wallet riderWallet = walletRepository.getWalletByRiderId(riderId);
            BigDecimal riderWalletAmount = riderWallet.getAmount();
            BigDecimal totalFare = paymentDTO.getTotalFare();

            BigDecimal newAmount = riderWalletAmount.subtract(totalFare);
            riderWallet.setAmount(newAmount);

            walletRepository.save(riderWallet);
        }

        return Payment.builder()
                .id(id)
                .paymentMode(paymentDTO.getPaymentMode())
                .bookingFee(paymentDTO.getBookingFee() != null ? paymentDTO.getBookingFee() : bookingFee)
                .discount(paymentDTO.getDiscount())
                .gst(paymentDTO.getGst())
                .rideCharge(paymentDTO.getRideCharge() != null ? paymentDTO.getRideCharge() : rideCharge)
                .totalFare(paymentDTO.getTotalFare())
                .paymentStatus(paymentDTO.getPaymentStatus())
                .paymentId(paymentDTO.getPaymentId())
                .nightCharges(paymentDTO.getNightCharges())
                .waitingCharges(paymentDTO.getWaitingCharges())
                .baseKmAmount(paymentDTO.getBaseKmAmount() != null ? paymentDTO.getBaseKmAmount() : baseKmAmount)
                .additionKmFee(paymentDTO.getAdditionKmFee() != null ? paymentDTO.getAdditionKmFee() : additionKmFee)
                .isPaymentInit(false)
                .isActive(true)
                .isDeleted(false)
                .build();
    }

    private DriverBookingAudit createDriverBookingAudit(Long tripId, Long driverId, String reason) {
        return DriverBookingAudit.builder()
                .trip(tripRepository.findById(tripId).get())
                .driver(driverRepository.findById(driverId).get())
                .reason(reason)
                .isActive(true)
                .isDeleted(false)
                .isDeclined(true)
                .build();
    }

    /**
     * @param driverId
     * @param tripId
     * @return
     */
    @Override
    public TripDTO acceptBooking(Long driverId, Long tripId) {
        Trip trip = tripRepository.findById(tripId)
                .orElseThrow(() -> new ResourceNotFoundException("Trip", "id", tripId));
        String deviceId = "";
        if (!trip.getTripStatus().equals(TripStatus.CANCELLED)) {
            if (!trip.getTripStatus().equals(TripStatus.BOOKED)) {
                if (trip.getDriver() == null) {
                    Driver driver = driverRepository.findById(driverId)
                            .orElseThrow(() -> new ResourceNotFoundException("Driver", "id", driverId));
                    trip.setDriver(driver);
                    trip.setTripStatus(TripStatus.BOOKED);
                    Rider rider = trip.getRider();
                    Trip save = tripRepository.save(trip);
                    if (trip.getTripStatus().equals(TripStatus.BOOKED)) {
                        if (rider.getDeviceId() != null) {
                            deviceId = rider.getDeviceId();
                        }
                        notificationService.saveNotification(convertNotificationForRider(trip, deviceId));
                        notifyTripBookedToRider(trip, deviceId);
                    }
                    return convertTrip(save);
                } else {
                    throw new AlreadyPresentException("Trip", "another", "Driver");
                }
            } else {
                throw new HimCabIndiaException("The Ride is Already BOOKED");
            }
        } else {
            throw new HimCabIndiaException("The Ride is CANCELLED");
        }

    }

    private Notification convertNotificationForRider(Trip trip, String deviceId) {
        Notification notification = new Notification();
        notification.setTrip(trip);
        notification.setDeviceId(deviceId);
        Rider rider = trip.getRider();
        notification.setRider(rider);
        notification.setName(rider.getFirstName() + " " + rider.getLastName());
        notification.setNotificationType(NotificationType.TRIP_ACCEPTED);
        notification.setTitle("Himcab");
        notification.setStatus(true);
        notification.setMessage(trip.getDriver().getFirstName() + " has accepted your ride");
        return notification;
    }

    private void notifyTripBookedToRider(Trip trip, String deviceId) {
        Rider rider = trip.getRider();
        Driver driver = trip.getDriver();
        Map<String, String> pushData = new HashMap<>();
        pushData.put("tripId", String.valueOf(trip.getId()));
        pushData.put("customerName", rider.getFirstName() + " " + rider.getLastName());
        pushData.put("notificationType", NotificationType.TRIP_ACCEPTED.toString());
        String driverName = driver.getFirstName();
        String message = driverName + " has accepted your ride";
        pushData.put("title", "Himcab");
        pushData.put("body", message);
        pushNotificationService.sendPushNotification(pushData, deviceId);
    }

    /**
     * @param id takes Long driver id.
     * @return {@link List<TripDTO>} .
     */

    @Override
    public Page<TripDTO> getDriverTrip(Long id, Pageable pageable, TripStatus tripStatus, PaymentMode paymentMode,
            TripType taxiType) {
        // When no specific tripStatus filter is provided, use the native query with proper status-based ordering
        if (tripStatus == null) {
            String taxiTypeStr = taxiType != null ? taxiType.name() : null;
            String paymentModeStr = paymentMode != null ? paymentMode.name() : null;
            Page<Trip> trips = tripRepository.getDriverTripsWithFilters(id, taxiTypeStr, paymentModeStr, pageable);
            return trips.map(this::convertTrip);
        }
        
        // When specific tripStatus is provided, use the existing Specification approach
        Specification<Trip> spec = Specification.where(null);
        if (id != null) {
            spec = spec.and((root, query, builder) -> builder.equal(root.get("driver").get("id"), id));
        }
        switch (tripStatus) {
            case COMPLETED:
            case CANCELLED:
                spec = spec.and((root, query, builder) -> builder.equal(root.get("tripStatus"), tripStatus));
                break;
            case ONGOING:
            case REACH_ON_DESTINATION:
            case REACH_ON_START_LOCATION:
                spec = spec.and((root, query, builder) -> builder.or(
                        builder.equal(root.get("tripStatus"), TripStatus.ONGOING),
                        builder.equal(root.get("tripStatus"), TripStatus.REACH_ON_DESTINATION),
                        builder.equal(root.get("tripStatus"), TripStatus.REACH_ON_START_LOCATION)));
                break;
            case UPCOMING:
            case BOOKED:
            case NOT_BOOKED:
                spec = spec.and((root, query, builder) -> builder.or(
                        builder.equal(root.get("tripStatus"), TripStatus.UPCOMING),
                        builder.equal(root.get("tripStatus"), TripStatus.BOOKED),
                        builder.equal(root.get("tripStatus"), TripStatus.NOT_BOOKED)));
                break;
        }
        if (paymentMode != null) {
            spec = spec
                    .and((root, query, builder) -> builder.equal(root.get("payment").get("paymentMode"), paymentMode));
        }
        if (taxiType != null) {
            spec = spec.and((root, query, builder) -> builder.equal(root.get("tripType"), taxiType));
        }
        pageable = PageRequest.of(pageable.getPageNumber(), pageable.getPageSize(),
                Sort.by(Sort.Direction.DESC, "scheduleDate"));
        Page<Trip> trips = tripRepository.findAll(spec, pageable);
        return trips.map(this::convertTrip);
    }

    private Direction convertDirectionDto(DirectionDTO dd) {
        return Direction.builder().address(dd.getAddress()).isActive(true).isDeleted(false).latitude(dd.getLatitude())
                .longitude(dd.getLongitude()).build();
    }

}