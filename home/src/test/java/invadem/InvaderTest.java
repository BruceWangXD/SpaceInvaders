package invadem;

import invadem.gameobject.Invader;
import invadem.gameobject.Projectile;
import invadem.App.*;
import org.junit.Test;
import static org.junit.Assert.*;


public class InvaderTest {

    @Test
    public void testInvaderConstruction() {
        Invader inv = new Invader(null,null,null,180,48,16,16,1,1,100);
        assertNotNull(inv);
    }

    @Test
    public void testInvaderTick() {
        Invader inv = new Invader(null,null,null,180,48,16,16,1,1,100);
        inv.rightTick();
        assertEquals(181,inv.getX());
        inv.leftTick();
        assertEquals(180,inv.getX());
        inv.downTick();
        assertEquals(49,inv.getY());
    }

    @Test
    public void testInvaderFireProjectile() {
        Invader inv = new Invader(null,null,null,180,48,16,16,1,1,100);
        assertNotNull(inv.fire());
    }

    @Test
    public void testInvaderAlive() {
        Invader inv1 = new Invader(null,null,null,180,48,16,16,1,1,100);
        assertEquals(true, inv1.alived());
    }

    @Test
    public void testInvaderNotAlive() {
        Invader inv = new Invader(null,null,null,180,48,16,16,1,1,100);
        inv.attacked();
        assertEquals(false, inv.alived());
    }

    @Test
    public void testInvaderIntersectWithPlayerProjectile() {
        Invader inv = new Invader(null,null,null,180,48,16,16,1,1,100);
        Projectile p = new Projectile(null,180,48,1,3,1,1 ,1);
        assertEquals(true, p.intersect(inv));

    }
    @Test
    public void testInvaderGetScore() {
        Invader inv = new Invader(null,null,null,180,48,16,16,1,1,100);
        assertEquals(100, inv.getScore());

    }

}
