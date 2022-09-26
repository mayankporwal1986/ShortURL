package com.porwalsoft.shorturl.controller;

import java.io.IOException;
import java.net.MalformedURLException;

import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import com.porwalsoft.shorturl.model.ShortURL;
import com.porwalsoft.shorturl.service.ShortURLService;

@RestController
public class ShortURLRestController {
	
	@Autowired
	ShortURLService sURLService;
	
	@RequestMapping(value="/shortenurl", method=RequestMethod.POST)
	public ResponseEntity<Object> getShortenUrl(@RequestBody ShortURL shortenUrl) throws MalformedURLException {
		String randomChar = sURLService.getRandomChars();
		sURLService.setShortUrl(randomChar, shortenUrl);
		return new ResponseEntity<Object>(shortenUrl, HttpStatus.OK);
	}
	
	@RequestMapping(value="/s/{randomstring}", method=RequestMethod.GET)
	public void getFullUrl(HttpServletResponse response, @PathVariable("randomstring") String randomString) throws IOException {
		response.sendRedirect(sURLService.shortURLList.get(randomString).getFull_url());
	}

}
