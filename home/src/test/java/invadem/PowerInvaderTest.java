package invadem;

import invadem.gameobject.Invader;
import invadem.gameobject.PowerInvader;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public class PowerInvaderTest {
    @Test
    public void testPowerInvaderConstruction() {
        PowerInvader inv = new PowerInvader(null,null,null,180,48,16,16,1,1,100);
        assertNotNull(inv);
    }
    @Test
    public void testPowerInvaderFire() {
        PowerInvader inv = new PowerInvader(null,null,null,180,48,16,16,1,1,100);
        assertNotNull(inv.fire());
    }
}
