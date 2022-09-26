package com.porwalsoft.shorturl.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

@Controller
public class ShortURLController {
	
	@RequestMapping(value = "/", method = RequestMethod.GET)
	public String loadIndex() {
		return "index";
	}

}
