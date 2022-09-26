package com.porwalsoft.shorturl.service;

import java.net.MalformedURLException;
import java.util.HashMap;
import java.util.Map;

import org.springframework.stereotype.Service;

import com.porwalsoft.shorturl.model.ShortURL;

@Service
public class ShortURLService {
	
	public Map<String, ShortURL> shortURLList = new HashMap<>();
	
	public String getRandomChars() {
		String randomStr = "";
		String possibleChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
		for (int i = 0; i < 5; i++)
			randomStr += possibleChars.charAt((int) Math.floor(Math.random() * possibleChars.length()));
		return randomStr;
	}
	
	public void setShortUrl(String randomChar, ShortURL shortenUrl) throws MalformedURLException {
		 shortenUrl.setShort_url("http://localhost:8080/s/"+randomChar);
		 shortURLList.put(randomChar, shortenUrl);
	}

}
