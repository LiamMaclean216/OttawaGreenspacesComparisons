<?php
	// Database parameters
    $servername = "";
    $dbname = "";
    $username = "";
    $password = "";
    $options = array(PDO::MYSQL_ATTR_SSL_CA => '../ssl/BaltimoreCyberTrustRoot.crt.pem');

    // Connection to database
    try {
        
        $conn = new PDO("mysql:host=$servername;port=3306;dbname=$dbname", $username, $password, $options);
        // Set the PDO error mode to exception
        $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        // echo "Connected successfully"; 
        }
    catch(PDOException $e)
        {
        // echo "Connection failed: " . $e->getMessage();
        }

    // Directory of images 
	$IMAGES_DIR = '../img/';
	$message = "";

	// Get values send by form
	$back = $_GET["back"];

	/**
    * Gets user's IP address
    * @return string IP address
    */
    function getUserIpAddr(){
        if(!empty($_SERVER['HTTP_CLIENT_IP'])){
            //ip from share internet
            $ip = $_SERVER['HTTP_CLIENT_IP'];
        }elseif(!empty($_SERVER['HTTP_X_FORWARDED_FOR'])){
            //ip pass from proxy
            $ip = $_SERVER['HTTP_X_FORWARDED_FOR'];
        }else{
            $ip = $_SERVER['REMOTE_ADDR'];
        }
        $ip_port = explode(":", $ip);
        $final_ip = $ip_port[0];
        return $final_ip;
    }
    $ip_address = getUserIpAddr();

	// If go back
	if ($back == 1) {

		// Get values send by form
		$img1_key = $_GET["previous_image_key1"];
		$img2_key = $_GET["previous_image_key2"];
		$img1_path = $_GET["previous_image_path1"];
		$img2_path = $_GET["previous_image_path2"];

		// Check values received from the form
	    $message = "";
	    $valid1 = preg_match('/[\w]+/', $img1_key);
	    $valid2 = preg_match('/[\w]+/', $img2_key);
	    $valid3 = (strlen($img1_key) == 26 && strlen($img2_key) == 26);
	    $valid4 = preg_match('/[\w]+/', $img1_path);
	    $valid5 = preg_match('/[\w]+/', $img2_path);
	    $valid6 = (strlen($img1_path) == 74 && strlen($img2_path) == 74);

	    if ($valid1 + $valid2 + $valid3 != 3){
	        $message .= "Unvalid keys\n";
	    }

	    if ($valid4 + $valid5 + $valid6 != 3){
	        $message .= "Unvalid paths\n";
	    }
	    // If values are ok make queries
	    if ($valid1 + $valid2 + $valid3 == 3){

			// Remove values from database
			$query = $conn->prepare("DELETE FROM duels_question_3 WHERE key1 = ? AND key2 = ?");
			$query->execute(array($img1_key, $img2_key));

			$query = $conn->prepare("UPDATE userswinter SET comparisons = comparisons - 1 WHERE ip_address = ?");
	    	$query->execute(array($ip_address));
	    }
	} else {
	    // Select values from database
		$query = "SELECT image_key, filename FROM imageswinter ORDER BY RAND() LIMIT 2";
	    $images = $conn->query($query)->fetchAll();

	    // Update values to be send 
	    $img1_key = $images[0]['image_key'];
		$img2_key = $images[1]['image_key'];

		$img1_path = $IMAGES_DIR . $images[0]['filename'];
		$img2_path = $IMAGES_DIR . $images[1]['filename'];
	}  
	
    // Send values to ajax
	$result = array('ip_address' => $ip_address,
		'back' => $back,
		'image_key1' => $img1_key,
		'image_key2' => $img2_key,
		'image_path1' => $img1_path,
		'image_path2' => $img2_path,
		'error_message' => $message);

	echo json_encode($result);

	// End connection to database
	$conn=null
?> 