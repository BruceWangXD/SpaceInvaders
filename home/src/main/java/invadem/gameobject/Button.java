package invadem.gameobject;

import processing.core.PApplet;
import processing.core.PImage;


public class Button extends GameObject {
    //declare different kinds of images
    private PImage imgOne;
    private PImage imgTwo;
    //constructor
    public Button(PImage imgOne, PImage imgTwo, int x, int y, int width, int height,int health,int velocity) {
        super(x, y, width, height,health,velocity);
        this.imgOne = imgOne;
        this.imgTwo = imgTwo;
    }
    //draw the button
    public void draw(PApplet app,boolean pressed) {
        if(!pressed){
            app.image(imgOne, x, y, width, height);
        }else{
            app.image(imgTwo, x, y, width, height);
        }
    }

}