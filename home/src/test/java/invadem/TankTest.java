package invadem;

import invadem.gameobject.*;
import invadem.App.*;
import org.junit.Test;
import processing.core.PApplet;
import processing.core.PImage;


import static org.junit.Assert.*;

public class TankTest {

    @Test
    public void testTankConstruction() {
        Tank tank = new Tank(null, null,309, 464, 22, 16, 3,1);
        assertNotNull(tank);
    }

    @Test
    public void testTankGet() {
        Tank tank = new Tank(null, null,309, 464, 22, 16, 3,1);
        assertEquals(309, tank.getX());
        assertEquals(464, tank.getY());
        assertEquals(22,tank.getWidth());
        assertEquals(16,tank.getHeight());
        assertEquals(3,tank.getHealth());
        assertEquals(1,tank.getVelocity());
    }

    @Test
    public void testTankProjectile() {
        Tank tank = new Tank(null, null,309, 464, 22, 16, 3,1);
        Projectile p = tank.fire(null);
        assertNotNull(p);
        assertEquals(320, p.getX());
        assertEquals(467, p.getY());
    }

    @Test
    public void testTankAlive() {
        Tank tank1 = new Tank(null, null,309, 464, 22, 16, 3,1);
        assertEquals(true, tank1.alived());
        Tank tank2 = new Tank(null, null,309, 464, 22, 16, 0,1);
        assertEquals(false, tank2.alived());
    }

    @Test
    public void testTankTick() {
        Tank tank = new Tank(null, null,309, 464, 22, 16, 3,1);
        tank.rightTick();
        assertEquals(310, tank.getX());
        tank.leftTick();
        assertEquals(309, tank.getX());
    }

    @Test
    public void testTankAttacked() {
        Tank tank = new Tank(null, null,309, 464, 22, 16, 3,1);
        Projectile p = new Projectile(null,309,464,1,3,1,1 ,1);
        tank.attacked(p);
        assertEquals(2, tank.getHealth());
        tank.attacked(p);
        tank.attacked(p);
        assertEquals(false, tank.alived());
    }


}
