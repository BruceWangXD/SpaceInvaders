package invadem.gameobject;

import processing.core.PImage;

public class PowerInvader extends Invader {
    //constructor
    public PowerInvader(PImage imgOne, PImage imgTwo, PImage projectileImage, int x, int y, int width, int height,int health, int velocity,int score) {
        super(imgOne,imgTwo, projectileImage, x, y, width, height, health,velocity,score);
    }
    //produce the power projectile
    public Projectile fire(){
        Projectile p = new Projectile(projectileImage,x+7,y-16,2,5,3,-1,3);
        return p;
    }
}